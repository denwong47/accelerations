# from concurrent.futures import ProcessPoolExecutor


import enum
import inspect
import math
from typing import Callable

import numpy as np
import pandas as pd
from numba import cuda
from tqdm import tqdm

from accelerations.tiler import tiler, \
                                tiler_coordinates, \
                                tiler_byte_operations, \
                                tiler_hashing

from accelerations.settings import DEFAULT_MEMORY_LIMIT
from accelerations.settings import DEBUG


# we can just use dtype().nbytes - this is just for reference
# dtype_nbytes = {
#     np.bool_: 1,
#     np.int8: 1,
#     np.complex128: 16,
#     np.complex256: 32,
#     np.complex64: 8,
#     np.float64: 8,
#     np.float16: 2,
#     np.float16: 2,
#     np.int64: 8,
#     np.int32: 4,
#     np.float128: 16,
#     np.longlong: 8,
#     np.int16: 2,
#     np.float32: 4,
#     np.uint8: 1,
#     np.uint64: 8,
#     np.uint32: 4,
#     np.ulonglong: 8,
#     np.uint16: 2,
# }

class AcceleratorTypeNotImplemented(NotImplementedError):
    def __nonzero__(self):
        return False
    __bool__=__nonzero__

class AcceleratedProcessInvalidInput(ValueError):
    def __nonzero__(self):
        return False
    __bool__=__nonzero__

class accelerator_type(enum.Enum):
    CPU_SINGLEPROCESS = "process_cpu"
    CPU_MULTIPROCESS = "process_cpu_parallel"
    CUDA = "process_cuda"
    OPENCL = "process_opencl"

    UNKNOWN = "process_cpu" # Default to single core CPU if accelerator type is not known


def cuda_available():
    try:
        _result = cuda.detect()
        return _result
    except cuda.cudadrv.error.CudaSupportError as e:
        return False

class accelerated_process():

    @classmethod
    def process(
        self,
        type:accelerator_type,
    ):
        _default = self.process_cpu_parallel
        if (isinstance(type, accelerator_type) and \
            cuda_available() if type is accelerator_type.CUDA else True):
            return getattr(self, type.value, _default)
        else:
            return _default

    def output_shape(**kwargs) -> tuple:
        if (kwargs.get("input1", None)):
            return kwargs["input1"].shape
        else:
            return AcceleratedProcessInvalidInput("No input provided; all input parameters needed to be keyworded.")

    # This is just for demonstration - not formally considered part of class
    def _process_factory(
        type_name:str,
    ):
        def _process(
            input1:np.ndarray,
            memory_limit:int=DEFAULT_MEMORY_LIMIT,
            **kwargs,
        ):
            return AcceleratorTypeNotImplemented(f"{type_name} process has not been implemented.")

        return _process

    # This process itself is not cuda.jit - it takes the arguments and process them with cuda.jit functions, leaving it completely transparent to the caller
    process_cpu = _process_factory("CPU single-threading")
    process_cuda = _process_factory("CPU multi-processing")
    process_opencl = _process_factory("OpenCL")
    process_cpu_parallel = _process_factory("CUDA")

    @staticmethod
    def tile_process(
        tiler_class:tiler,
        show_progress:bool=DEBUG,
    ):
        # Decorator Factory
        def _decorator(func):
            def _wrapper(*args, **kwargs):

                memory_limit = kwargs.get("memory_limit", DEFAULT_MEMORY_LIMIT)

                _tiler = tiler_class(
                    inputs=kwargs,
                    outputs=None,
                    memory_limit=memory_limit,
                    show_progress=show_progress,
                )

                if (show_progress):
                    print (f"Memory limit set to {memory_limit:,d} bytes.")
                    print (f"Tile shape: {_tiler.tile_shape}")
                    _pbar = tqdm(total=_tiler.no_of_tiles)
                    _pbar.update(1)

                _output = _tiler.outputs
                _factory = _tiler.tiles()

                _tiled_input = next(_factory)
                while (_tiled_input := _factory.send(func(*args, **_tiled_input))) is not None:
                    if (show_progress):
                        _pbar.set_description(f"Processing tile #{_tiler.counter}...")
                        _pbar.update(1)

                if (show_progress):
                    _pbar.close()

                if (len(_output) == 1):
                    return _output[0]
                else:
                    return _output

            return _wrapper
        return _decorator

