import abc
import datasets
import itertools
import torch
from typing import Callable, Iterator
from torchtyping import TensorType as TT

from training_interpretability.tokenizer import Tokenizer


class BatchLoader(abc.ABC):

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError


class PileLoader(BatchLoader):
    """
    Provides a batch iterator into the Pile dataset using a given tokenizer.

    Note that the Pile was recently taken offline due to copyright concerns,
    so we use an alternative provided by Monology that excludes copyrighted
    material.
    """

    HUGGINGFACE_DATASET="monology/pile-uncopyrighted"

    def __init__(self, batch_size: int, context_length: int, tokenizer: Tokenizer, split_type: str="train", device: str="cuda"):
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size parameter must be a positive integer"
        assert isinstance(context_length, int) and context_length > 0, "context_length parameter must be a positive integer"
        assert split_type in ("train", "test"), "split_type must be 'train' or 'test'"
        assert isinstance(tokenizer, Tokenizer), "tokenizer must be a Tokenizer"
        assert device in ("cuda", "cpu"), "device must be 'cuda' or 'cpu'"

        self.batch_size = batch_size
        self.context_length = context_length
        self.split_type = split_type
        self.tokenizer = tokenizer
        self.data = None
        self.device = device

    def __iter__(self) -> Iterator[TT["batch_size", "context_length"]]:
        self.data = iter(self._tokenize(self._load()))
        self._next_tokens = None
        return self

    def _load(self) -> datasets.iterable_dataset.IterableDataset:
        return datasets.load_dataset(PileLoader.HUGGINGFACE_DATASET, split=self.split_type, streaming=True)

    def _tokenize(self, data: datasets.iterable_dataset.IterableDataset) -> datasets.iterable_dataset.IterableDataset:
        data = data.map(lambda x: {"text": self.tokenizer(x["text"])})
        assert isinstance(data, datasets.iterable_dataset.IterableDataset)
        return data

    def __next__(self) -> TT["batch_size", "context_length"]:
        if not self.data:
            self.__iter__()

        offset = 0
        out = None
        try:
            if self._next_tokens is not None:
                tokens = self._next_tokens
            else:
                tokens = torch.tensor(next(self.data)["text"], dtype=torch.long, device=self.device)
            out = torch.zeros(self.batch_size * self.context_length, dtype=torch.long, device=tokens.device)
            while offset < out.shape[0]:
                if tokens.shape[0] <= out.shape[0] - offset:
                    out[offset:offset + tokens.shape[0]] = tokens
                    offset += tokens.shape[0]
                    if offset == out.shape[0]:
                        self._next_tokens = None
                    else:
                        tokens = torch.tensor(next(self.data)["text"], dtype=torch.long, device=self.device)
                else:
                    out[offset:] = tokens[:out.shape[0] - offset]
                    self._next_tokens = tokens[(out.shape[0] - offset):]
                    offset = out.shape[0]

        except StopIteration:
            if out is None:
                raise
            else: # We ran out of data mid-way through a batch.
                # Pad the last record with the eos token, presumably at the 
                # end of the previous last record.
                out[offset:] = out[offset-1]

        return out.view((self.batch_size, self.context_length))

