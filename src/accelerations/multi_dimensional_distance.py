import math

from numba.np.ufunc import parallel
import numpy as np
import numba
from numba import cuda

from accelerations.accelerator import accelerated_process, \
                                      accelerator_type, \
                                      AcceleratedProcessInvalidInput
from accelerations.tiler import tiler_coordinates
from accelerations.settings import DEFAULT_MEMORY_LIMIT, CUDA_DEFAULT_BLOCK_DIM
from accelerations.settings import DEBUG, DEBUG_TILER

# ========================================================================================

def distance_between_two_points(
        coor1:np.ndarray,
        coor2:np.ndarray,
        dimensions:np.int64,
    ):
        diff = coor2[:dimensions]-coor1[:dimensions]
        _distance = np.sqrt(np.sum(np.square(diff)))

        return _distance


# ========================================================================================

njit_distance_between_two_points = numba.njit(distance_between_two_points)

# ========================================================================================

# CUDA does not allow numpy array functions; so we have to loop through all the arrays.
@cuda.jit(device=True)
def cuda_distance_between_two_points(
        coor1:np.ndarray,
        coor2:np.ndarray,
        dimensions:np.int64,
    ):
        _cumulated = 0
        for _dim in range(coor1.shape[0]):
            _cumulated += (coor2[_dim] - coor1[_dim]) ** 2

        _distance = _cumulated ** 0.5

        return _distance

@cuda.jit
def cuda_distance_between_arrays(
    input1:np.ndarray,
    input2:np.ndarray,
    dimensions:np.int64,
    output:np.ndarray,
):

    # When running a kernel, the kernel function’s code is executed by every thread once.
    # It therefore has to know which thread it is in, in order to know which array element(s) it is responsible for (complex algorithms may define more complex responsibilities, but the underlying principle is the same).
    x, y = cuda.grid(2)
    
    # The above is a shorthand for below:
    # This ignores all the complexity of trying to work out the location of the block etc.
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    # bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y
    # bw = cuda.blockDim.x
    # bh = cuda.blockDim.y

    if x < input1.shape[0] and y < input2.shape[0]:
        dist = cuda_distance_between_two_points(
            input1[x],
            input2[y],
            dimensions,
        )
        
        output[x,y] = dist
    else:
        return

# ========================================================================================

class multi_dimensional_distance(accelerated_process):

    def output_shape(
        input1:np.ndarray,
        input2:np.ndarray,
        dtype:type=np.double,
        **kwargs,
    ) -> tuple:
        if (isinstance(input1, np.ndarray) and \
            isinstance(input2, np.ndarray)):
            return (
                input1.shape[0],
                input2.shape[0],
            )
        else:
            return AcceleratedProcessInvalidInput("multi_dimensional_distance require both input1 and input2. All input parameters needed to be keyworded.")
    
    def process_cpu(
        input1:np.ndarray,
        input2:np.ndarray,
        dtype:type=np.double,
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        **kwargs,
    ) -> np.ndarray:
        _no_of_dimensions = min(input1.shape[1], input2.shape[1])

        output = np.empty((input1.shape[0], input2.shape[0]), dtype=np.double)

        # This looks n^2 but with prange this will essentially be vectorized
        for i in numba.prange(input1.shape[0]):
            for j in numba.prange(input2.shape[0]):
                output[i,j] = njit_distance_between_two_points(input1[i,:_no_of_dimensions], input2[j,:_no_of_dimensions], _no_of_dimensions)

        return output
    
    # For Parallel, we just need to @numba.njit it
    process_cpu_parallel = numba.njit(process_cpu, parallel=True)
    process_cpu = numba.njit(process_cpu)

    @accelerated_process.tile_process(
        tiler_class=tiler_coordinates,
        show_progress=DEBUG_TILER, #use DEBUG if you want to debug tiler
        )
    def process_cuda(
        input1:np.ndarray,  # Do not rename - it has to be called input\d
        input2:np.ndarray,  # Do not rename - it has to be called input\d
        dtype:type=np.double,
        block_dim:tuple=CUDA_DEFAULT_BLOCK_DIM,
        memory_limit:int=DEFAULT_MEMORY_LIMIT, # decorator will use this via kwargs["memory_limit"] - do not delete
        show_progress:bool=False,
        **kwargs,
    ) -> np.ndarray:
        
        grid_dim = (
            math.ceil(input1.shape[0]/block_dim[0]),
            math.ceil(input2.shape[0]/block_dim[1])
        )

        _no_of_dimensions = min(input1.shape[1], input2.shape[1])

        output = np.empty((input1.shape[0], input2.shape[0]), dtype=dtype)

        _ondevice_input1 = cuda.to_device(np.ascontiguousarray(input1[:,:_no_of_dimensions]))
        _ondevice_input2 = cuda.to_device(np.ascontiguousarray(input2[:,:_no_of_dimensions]))
        _ondevice_output = cuda.to_device(output)
        
        if (show_progress):
            print ("DATA   DIM : (%d, %d)" % (input1.shape[0], input2.shape[0]))
            print (" : ".join(["GRID   DIM", str(grid_dim)]))
            print (" : ".join(["BLOCK  DIM", str(block_dim)]))

        cuda_distance_between_arrays[grid_dim, block_dim](
            _ondevice_input1,
            _ondevice_input2,
            _no_of_dimensions,
            _ondevice_output
        )

        _ondevice_output.copy_to_host(output)

        return output

    process_opencl = process_cpu_parallel