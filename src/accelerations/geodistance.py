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
from accelerations.settings import DEBUG, DEBUG_TILER

# ========================================================================================

def geodistance_between_two_latlngs(
    s_lat:np.float64,
    s_lng:np.float64,
    e_lat:np.float64,
    e_lng:np.float64,
    ):
    # Haversine calculation with CUDA acceleration.
    # Copied from https://towardsdatascience.com/better-parallelization-with-numba-3a41ca69452e
    
    # approximate radius of earth in km
    R = 6373.0
    
    # Discard calculation if latitude doesn't make any sense
    if (-90 <= s_lat <= 90 and -90 <= e_lat <= 90):
    
        s_lat = s_lat * math.pi / 180                     
        s_lng = s_lng * math.pi / 180 
        e_lat = e_lat * math.pi / 180                    
        e_lng = e_lng * math.pi / 180

        d = math.sin((e_lat - s_lat)/2)**2 + \
            math.cos(s_lat)*math.cos(e_lat) * \
                math.sin((e_lng - s_lng)/2)**2

        return 2 * R * math.asin(math.sqrt(d))
    else:
        # Returning NaN to cuda_distance_array means irrelevance
        return np.nan

def geodistance_between_arrays(
        coor1:np.ndarray,
        coor2:np.ndarray,
        dimensions:np.int64,
    ):
    pass
        # diff = coor2[:dimensions]-coor1[:dimensions]
        # _distance = np.sqrt(np.sum(np.square(diff)))

        # return _distance


# ========================================================================================

njit_geodistance_between_two_latlngs = numba.njit(geodistance_between_two_latlngs)
njit_geodistance_between_two_arrays = numba.njit(geodistance_between_arrays)

# ========================================================================================


cuda_geodistance_between_two_latlngs = cuda.jit(device=True)(geodistance_between_two_latlngs)

@cuda.jit
def cuda_geodistance_between_arrays(
    input1:np.ndarray,
    input2:np.ndarray,
    max_dist:np.float64,
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

    #x = tx + bx * bw
    #y = ty + by * bh

    #bpg = max(cuda.gridDim.x, cuda.gridDim.y)    # blocks per grid

    if x < input1.shape[0] and y < input2.shape[0]:

        lat_filter = max_dist / 100

        # escape condition if latitudes are too far apart
        if (math.fabs(input1[x,0] - input2[y,0]) <= lat_filter or max_dist < 0):
            dist = cuda_geodistance_between_two_latlngs(input1[x,0], input1[x,1],
                                                        input2[y,0], input2[y,1])

            # TEST ONLY - OUT OF BOUND cuda.blockIdx.y BEING GIVEN
            # See if-else note below
            #dist = bx + 0.01* by
            #dist = tx + 0.01* ty

            # if exceeds max_dist, replace with NaN
            # this is to allow for max_dist being used for purposes such as city radius or nodes grouping etc.
            if (dist > max_dist and max_dist>0):
                dist = np.nan
        else:
            dist = np.nan

        output[x, y] = dist
    else:
        # Quit if (x, y) is outside of valid boundary
        
        # Suspect that return may not work at all times - best to capture the entire operation within an if-else.
        return

# ========================================================================================
