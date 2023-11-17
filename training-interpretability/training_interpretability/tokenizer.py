import abc
import torch
from typing import List, Union

import transformers


class Tokenizer(abc.ABC):

    @abc.abstractmethod
    def encode(self, input: str) -> List[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, input: List[int]) -> List[str]:
        raise NotImplementedError
    
    def __call__(self, input: Union[str, List[int]]) -> Union[str, List[int]]:
        if isinstance(input, str):
            return self.encode(input)
        else:
            return self.decode(input)

class GPT2Tokenizer(Tokenizer):

    def __init__(self, gpt_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        self.gpt_tokenizer = gpt_tokenizer
        self.eos_token = self.gpt_tokenizer.encode(self.gpt_tokenizer.eos_token)[0]
    
    def encode(self, input: str, append_eos_token: bool=True) -> List[int]:
        output = self.gpt_tokenizer.encode(input)
        if append_eos_token:
            output.append(self.eos_token)
        return output
    
    def decode(self, input: List[int]) -> str:
        return self.gpt_tokenizer.decode(input)


