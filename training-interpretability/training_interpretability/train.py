from dataclasses import dataclass
import math
import os
import time
import torch
from training_interpretability.config import CheckpointModelConfig
from training_interpretability.data import BatchLoader
from training_interpretability.gradient_recorder import GradientRecorder
from training_interpretability.logger import Logger, StdoutLogger
from training_interpretability.tokenizer import Tokenizer
from training_interpretability.utils import with_retry
from typing import Optional, Tuple

class Trainer:
    """
    Coordinator for training with gradient capture.

    The Trainer class is responsible for:
    - Maintaining hyperparameters relevant to a training run.
    - Reading from a DataLoader.
    - Providing the data from a DataLoader to a Model.
    - Tracking metadata and performing additional activity during the run.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer, data_loader: BatchLoader,
                 train_steps: int, d_vocab: int,
                 lr: float=10e-3, lr_step: int=1000, lr_gamma: float=0.03,
                 clip_grad_norm: Optional[float]=0.5,
                 gradient_recorder: Optional[GradientRecorder]=None,
                 logger: Logger=StdoutLogger,
                 checkpoint_model_config: Optional[CheckpointModelConfig]=None):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader

        # Logging related attributes
        self.logger = logger() if isinstance(logger, type) else logger
        self.log_interval = 100

        # Gradient recording related attributes
        self.gradient_recorder = gradient_recorder
        self.checkpoint_model_config = checkpoint_model_config

        # Learning-related parameters
        self.train_steps = train_steps
        self.d_vocab = d_vocab
        self.lr, self.lr_step, self.lr_gamma = lr, lr_step, lr_gamma
        self.clip_grad_norm = clip_grad_norm
     
    def _init_train_attributes(self, reshape: Tuple[int, ...], device: str):
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.lr_step, gamma=self.lr_gamma)

        if self.gradient_recorder:
            # Some gradient recording options need to modify the loss criterion.
            self.gradient_recorder.replace_loss_criterion(self.loss_criterion, reshape, device)

    def train(self, reshape: Tuple[int, ...], device: str):
        """
        Perform full training run for a model.
        """
        self._init_train_attributes(reshape, device)
        self.model.train()

        self._train_epoch()
    
    def _train_epoch(self):
        self.logger("Starting model training\n")
        self._init_log_loss()
        for batch_ix in range(self.train_steps):
            self._total_loss += self._train_step(batch_ix)

            if batch_ix % self.log_interval == 0 and batch_ix > 0:
                self._log_loss(batch_ix)

    def _train_step(self, batch_ix: int) -> float:
        """
        Run one train step and return train loss.

        :batch_ix int Index of the current batch.
        """
        
        data_batch = next(self.data_loader)

        if self.gradient_recorder:
            cur_lr = self.scheduler.get_last_lr()[0]
            loss = self.gradient_recorder.train_step(batch_ix, self.model, self.loss_criterion,
                                            self.optimizer, cur_lr, data_batch,
                                            clip_norm=self.clip_grad_norm)

            self.scheduler.step()
            self._checkpoint_model_if_necessary(batch_ix)
            return loss
        else:
            output = self.model(data_batch)
            loss = self.loss_criterion(output.view(-1, self.d_vocab)[:-1, :], data_batch.view(-1)[1:])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self._checkpoint_model_if_necessary(batch_ix)
            return loss.item()

    def _init_log_loss(self):
        if not hasattr(self, "_start_time"):
            self._start_time = time.time()
        if not hasattr(self, "_total_loss"):
            self._total_loss = 0
    
    def _log_loss(self, batch_ix: int):
        self._init_log_loss()

        lr = self.scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - self._start_time) * 1_000 / self.log_interval
        cur_loss = self._total_loss / self.log_interval
        try:
            ppl = math.exp(cur_loss)
        except:
            ppl = -1.0

        self.logger.progress(
            f"epoch {0:3d}",
            f"{batch_ix:5d}/{self.train_steps:5d} batches",
            f"lr {lr:02.02f}",
            f"ms/batch {ms_per_batch:5.2f}",
            f"loss {cur_loss:5.2f}",
            f"ppl {ppl:8.2f}\n",
            wrapper=False
        )

        self._total_loss = 0
        self._start_time = time.time()
    
    @with_retry(attempts=3, backoff=(60, 180, 360))
    def _checkpoint_model_if_necessary(self, train_step: int):
        if self.checkpoint_model_config and train_step % self.checkpoint_model_config.checkpoint_interval == 0:
            if not os.path.exists(self.checkpoint_model_config.checkpoint_dir):
                os.makedirs(self.checkpoint_model_config.checkpoint_dir, exist_ok=True)
            model_params_path = os.path.join(self.checkpoint_model.checkpoint_dir, f"{train_step}.pt")
            torch.save(self.model.state_dict(), model_params_path)
