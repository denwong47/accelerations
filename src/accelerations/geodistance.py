import math

from numba.np.ufunc import parallel
import numpy as np
import numba
from numba import cuda

from accelerations.accelerator import accelerated_process, \
                                      accelerator_type, \
                                      output_matrix_type, \
                                      AcceleratedProcessInvalidInput
from accelerations.settings import DEFAULT_MEMORY_LIMIT, CUDA_DEFAULT_BLOCK_DIM
from accelerations.settings import DEBUG