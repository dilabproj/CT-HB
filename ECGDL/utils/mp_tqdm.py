"""MP_TQDM v4 by Micro

This module is a wrapper for easy multi-processing with tqdm
"""

import math
import itertools
import multiprocessing

from functools import wraps
from typing import List, Dict, Any, Callable, Iterable, Optional

from tqdm.auto import tqdm


ParamList = Dict[str, Any]


def mp_tqdm(func: Callable,
            args: Iterable[ParamList],
            args_len: Optional[int] = None,
            shared: Optional[ParamList] = None,
            task_size: int = 1,
            process_cnt: int = 1,
            ordered: bool = False,
            reset: bool = True) -> List[Any]:
    """This function multi-processes the workload

    Args:
        func: A function that is decorate by MP_TQDM_WORKER
        args: Iterable of Parameters for each task
        args_len: Length of Iterable of Parameters
        shared: Shared Parameters for each task
        task_size: Size of a single batch
        process_cnt: Number of worker processes
        ordered: Return the output in order
        reset: Do workers need to be reset between batches

    Returns:
        Returns a List of Function Returns which is not in order of original Args
    """
    def grouper(iterable, n):
        iterable = iter(iterable)

        def add_param(x):
            return (process_cnt, shared, x)

        return iter(lambda: add_param(list(itertools.islice(iterable, n))), add_param([]))

    rets: List[Any] = []
    with multiprocessing.Pool(process_cnt, maxtasksperchild=1 if reset else None) as p:
        # The master process tqdm bar is at Position 0
        if args_len is None:
            try:
                args_len = len(args)  # type: ignore
            except Exception:
                args_len = None
        total_chunks = None if args_len is None else math.ceil(args_len / task_size)
        mapmethod = p.imap if ordered else p.imap_unordered
        for ret in tqdm(mapmethod(func, grouper(args, task_size)),
                        total=total_chunks, dynamic_ncols=True):
            rets += ret
    return rets


def mp_tqdm_worker(func: Callable) -> Callable:
    """This is a decorator function to decorate worker functions

    Args:
        Callable: A Callable that takes in shared args and a single task in list of args
            and do necessary processing before returning results

    Note:
        Do not include tqdm in worker callable

    Returns:
        Returns a List of Function Returns which is in order of original Args
    """
    @wraps(func)
    def d_func(args):
        process_cnt, shared, argset = args
        shared = shared if shared is not None else {}
        # pylint: disable=protected-access
        worker_id = (multiprocessing.current_process()._identity[0] - 1) % process_cnt + 1
        # pylint: enable=protected-access
        rets = []
        for arg in argset:
            rets.append(func(worker_id=worker_id, **shared, **arg))
        return rets
    return d_func
