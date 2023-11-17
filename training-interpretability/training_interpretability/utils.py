from functools import reduce, wraps
import time
import torch
from typing import Tuple, Union


def with_retry(attempts: int, backoff=Union[int, Tuple[int, ...]]):
    def decorator(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == attempts - 1:
                        raise e
                    else:
                        retry_time = backoff[attempt] if isinstance(backoff, tuple) else backoff
                        print(f"Exception encountered, retrying in {retry_time}s: {e}")
                        time.sleep(retry_time)

        return wrapper
    
    return decorator

def fetch_model_module_from_parameter_key(model: torch.nn.Module, parameter_key: str) -> torch.nn.Module:
    """
    Converts a parameter key to a model module.

    Example:
        >> fetch_module_from_parameter_key(model, 'blocks.0.ln1.w') is model.blocks[0].ln1.w
    """
    def fetch_component(module: torch.nn.Module, key: str) -> torch.nn.Module:
        if key[0] in "0123456789":
            return module[int(key)]
        else:
            return getattr(module, key)

    return reduce(fetch_component, parameter_key.split("."), model)
