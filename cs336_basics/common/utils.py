import functools
from typing import Callable

import torch


def fp32_precision(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        idx_to_dtype = {}
        new_args, new_kwargs = [], {}

        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.dtype != torch.float32:
                idx_to_dtype[idx] = arg.dtype
                arg = arg.to(torch.float32)
            new_args.append(arg)

        for idx, kwarg in kwargs.items():
            if isinstance(kwarg, torch.Tensor) and kwarg.dtype != torch.float32:
                idx_to_dtype[idx] = kwarg.dtype
                kwarg = kwarg.to(torch.float32)
            new_kwargs[idx] = kwarg

        result = func(*new_args, **new_kwargs)

        for idx, dtype in idx_to_dtype.items():
            if isinstance(result, torch.Tensor):
                result = result.to(dtype)
            else:
                result[idx] = result[idx].to(dtype)

        return result

    return wrapper
