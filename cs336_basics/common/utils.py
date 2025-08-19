from collections.abc import Callable, Iterable
import functools

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

        # TODO(nino): change the logic for setting back the dtype
        for idx, dtype in idx_to_dtype.items():
            if isinstance(result, torch.Tensor):
                result = result.to(dtype)
            else:
                result[idx] = result[idx].to(dtype)

        return result

    return wrapper


def require_param(param_names: Iterable[str]) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for p_name in param_names:
                if p_name not in kwargs:
                    raise ValueError(f"{p_name} is required for {func.__name__}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
