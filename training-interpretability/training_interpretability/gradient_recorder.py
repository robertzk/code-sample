import abc
import os
import time
import torch
import tqdm
from torchtyping import TensorType as TT
from typing import Any, Dict, Iterable, Optional, Tuple

from training_interpretability.config import CheckpointModelConfig
from training_interpretability.utils import with_retry, fetch_model_module_from_parameter_key


class CrossEntropyLossDimMean(torch.nn.Module):
    """
    Wraps cross-entropy loss without reduction to compute the mean of the loss
    over a given dimension.
    """

    def __init__(self, loss_criterion: torch.nn.CrossEntropyLoss, reshape: Tuple[int, ...], device: str, dim: int=-1):
        """
        Arguments:
            loss_criterion: Cross entropy loss criterion to mean reduce over.
            reshape: The dimension to reshape the tensor into.
            device: Device of padding token.
            dim: Dimension to mean reduce over. Default: -1
        """
        if not isinstance(loss_criterion, torch.nn.CrossEntropyLoss):
            raise ValueError("CrossEntropyLossDimMean can only wrap a CrossEntropyLoss.")
        assert isinstance(dim, int)
        super().__init__()
        
        self.loss_criterion = loss_criterion
        self.reshape = reshape
        self.device = device
        self.dim = dim
    
    def forward(self, *args):
        if not hasattr(self, "_pad"):
            # TODO: We may want to replace this with a mean of all the grads to avoid
            # bias on the last datum.
            self._pad = torch.tensor([0], device=self.device)

        if self.loss_criterion.reduction == "none":
            loss = torch.cat((self.loss_criterion.forward(*args), self._pad)).view(*self.reshape)
            return loss.mean(dim=self.dim)
        else:
            return self.loss_criterion.forward(*args)

class GradientRecorder(abc.ABC):

    def __init__(self, log_single_datums: bool=False, exclude_units: Iterable[str]=tuple(),
                 checkpoint_model_config: Optional[CheckpointModelConfig]=None):
        """
        :exclude_units Iterable[str] A list of units to exclude (e.g. "encoder.weight")
        """
        self.log_single_datums = log_single_datums
        self.exclude_units = exclude_units

        self.checkpoint_model_config = checkpoint_model_config

    def init_model(self, model: torch.nn.Module):
        for parameter in model.parameters():
            parameter.requires_grad = True

    @abc.abstractmethod
    def record(self, batch_ix: int, datum_ix: int, gradients: Dict[str, torch.Tensor], data: TT["batch_size", "context_length"]):
        raise NotImplementedError
    
    @abc.abstractmethod
    def train_step(self, batch_ix: int, model: torch.nn.Module, loss_criterion: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, lr: float, data: TT["batch_size", "context_length"]) -> float:
        raise NotImplementedError

    def replace_loss_criterion(self, loss_criterion: Any, reshape: Tuple[int, ...], device: str):
        if self.log_single_datums and isinstance(loss_criterion, torch.nn.CrossEntropyLoss):
            # To capture individual datum gradients, we need to set reduction="none".
            # In fact, this will provide token-level gradients.
            loss_criterion.reduction = "none"
            self._loss_criterion = CrossEntropyLossDimMean(loss_criterion, reshape, device)
            

class BasicGradientRecorder(GradientRecorder):

    def __init__(self, log_single_datums: bool=False, exclude_units: Iterable[str]=tuple(),
                 apply_grad_sanity_check: bool=False,
                 checkpoint_model_config: Optional[CheckpointModelConfig]=None):
        super().__init__(log_single_datums, exclude_units, checkpoint_model_config)
        self.apply_grad_sanity_check = apply_grad_sanity_check 

    def train_step(self, batch_ix: int, model: torch.nn.Module, loss_criterion: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   lr: float, data: TT["batch_size", "context_length"],
                   clip_norm: Optional[float]=None) -> float:

        if not hasattr(self, "_model_copy"):
            self._model_copy = type(model)(model.cfg)

        output = model(data)
        loss = self._loss_criterion(output.view(-1, model.cfg.d_vocab)[:-1, :], data.view(-1)[1:])

        total_grads = {
            name: torch.zeros_like(param) for name, param in model.named_parameters()
        }

        # At this point, the loss has been flattened to show data contiguously by batch.
        # We iterate over each batch and context to process the gradients into datum-level gradients.
        for batch in tqdm.tqdm(range(data.shape[0])):
            new_grads = {
                name: torch.zeros_like(param) for name, param in model.named_parameters()
                if name not in self.exclude_units
            }

            loss[batch].backward(retain_graph=True)

            for name, param in model.named_parameters():
                if name not in self.exclude_units:
                    new_grads[name].add_(param.grad.data * -lr / loss.size(0))
                total_grads[name].add_(param.grad.data)
        
            model.zero_grad()

            self.record(batch_ix, batch, new_grads, data)
        
        del new_grads
        if data.device == "cuda":
            torch.cuda.empty_cache()
        
        # First we apply the gradient averaged over all batches and tokens,
        # then we clip norms if required.
        for name, param in model.named_parameters():
            param.grad = total_grads[name] / loss.size(0)
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        def sanity_check(loss_size):
            # Verify that the individual summed grads are equal
            # to a single pass with a CrossEntropyLoss(reduction="mean").
            model.zero_grad()
            try:
                reduction = self._loss_criterion.loss_criterion.reduction
                self._loss_criterion.loss_criterion.reduction = "mean"
                output = model(data)
                loss2 = self._loss_criterion(output.view(-1, output.shape[-1])[:-1], data.view(-1)[1:])
                loss2.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                params = { name: param for name, param in model.named_parameters() }
                assert sum(
                    1 if ((params[name].grad - total_grads[name] / loss_size).abs() < 9e-2).all() else 0 
                    for name, _ in model.named_parameters()
                    if not name in self.exclude_units
                ) / len(params) > 0.85
            finally:
                self._loss_criterion.loss_criterion.reduction = "none"

        if self.apply_grad_sanity_check:
            sanity_check(loss.size(0))
        
        optimizer.step()
        cur_loss = loss.mean().item()

        # Clean up some more GPU mem usage explicitly.
        del loss
        del total_grads
        if data.device == "cuda":
            torch.cuda.empty_cache()
        
        if hasattr(self, "_model_copy"):
            del self._model_copy
        
        return cur_loss

class BasicFileGradientRecorder(BasicGradientRecorder):

    def __init__(self, log_single_datums: bool=False, exclude_units: Iterable[str]=tuple(),
                 apply_grad_sanity_check: bool=False, dir_path: str="grads",
                 checkpoint_model_config: Optional[CheckpointModelConfig]=None):
        super().__init__(log_single_datums, exclude_units, apply_grad_sanity_check, checkpoint_model_config)
        self.apply_grad_sanity_check = apply_grad_sanity_check 
        self.dir_path = dir_path

    @with_retry(attempts=3, backoff=(60, 180, 360))
    def record(self, batch_ix: int, datum_ix: int, gradients: Dict[str, torch.Tensor],
               data: TT["batch_size", "context_length"]):
        dir = os.path.join(self.dir_path, str(batch_ix), str(datum_ix))

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        for name, _ in self._model_copy.named_parameters():
            *keys, last = name.split(".")
            if name in gradients:
                setattr(fetch_model_module_from_parameter_key(self._model_copy, ".".join(keys)), last, torch.nn.Parameter(gradients[name]))
            else:
                setattr(fetch_model_module_from_parameter_key(self._model_copy, ".".join(keys)), last, None)
        torch.save(self._model_copy.state_dict(), os.path.join(dir, "grads.pt"))
        torch.save(data, os.path.join(dir, "data.pt"))
