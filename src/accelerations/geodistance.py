import math

from numba.np.ufunc import parallel
import numpy as np
import numba
from numba import cuda

try:
    import pyopencl as cl
except ModuleNotFoundError:
    cl = None

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

# COPIED OVER FROM OLD GEODISTANCE MODULE - NEEDS ADAPTING
def opencl_distance_array(coordinate_set1, coordinate_set2, ctx, queue, max_dist=None):
    
    width_np = np.int32(coordinate_set1.shape[0])

    mf = cl.mem_flags
    s_lng_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(coordinate_set1[:,0]).astype(np.float32))
    s_lat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(coordinate_set1[:,1]).astype(np.float32))
    e_lng_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(coordinate_set2[:,0]).astype(np.float32))
    e_lat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(coordinate_set2[:,1]).astype(np.float32))
    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=width_np)

    # s_lat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.repeat(0.5,10))
    # s_lng_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.repeat(0.5,10))
    # e_lat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.repeat(2,10))
    # e_lng_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.repeat(2,10))


    kernel = """
    float cl_distance_between(float s_lat, float s_lng, float e_lat, float e_lng);

    float cl_distance_between(float s_lat, float s_lng, float e_lat, float e_lng){
        float R = 6373.0F;

        if (s_lat >= -90.0F && s_lat <= 90.0F && e_lat >= -90.0F && e_lat <= 90.0F)
        {
            float s_lat_rad = s_lat * M_PI_F / 180;
            float s_lng_rad = s_lng * M_PI_F / 180;
            float e_lat_rad = e_lat * M_PI_F / 180;
            float e_lng_rad = e_lng * M_PI_F / 180;

            float distance = pow(sin((e_lat_rad - s_lat_rad)/2), 2) + cos(s_lat_rad) * cos(e_lat_rad) * pow(sin((e_lng_rad - s_lng_rad)/2), 2);

            return 2 * R * asin(sqrt(distance));
        } else {
            return -1;
        }
    }


    __kernel void cl_distance_array(
        __global const float *s_lat_g,
        __global const float *s_lng_g, 
        __global const float *e_lat_g,
        __global const float *e_lng_g,
        __global const unsigned int *width,
        __global float *res_g)
    {
        size_t global_id_0 = get_global_id(0);
        size_t global_id_1 = get_global_id(1);
        size_t global_size_0 = get_global_size(0);

        size_t gid = global_id_1 * global_size_0 + global_id_0;

        res_g[gid] = cl_distance_between(s_lat_g[global_id_1], s_lng_g[global_id_1], e_lat_g[global_id_0], e_lng_g[global_id_0]);
    }

    __kernel void id_check(
        __global const float *s_lat_g,
        __global const float *s_lng_g, 
        __global const float *e_lat_g,
        __global const float *e_lng_g,
        __global const unsigned int *width,
        __global int *res_g)
    {
        size_t global_id_0 = get_global_id(0);
        size_t global_id_1 = get_global_id(1);
        size_t global_size_0 = get_global_size(0);
        //size_t offset_0 = get_global_offset(0);
        //size_t offset_1 = get_global_offset(1);

        //int index_0 = global_id_0 - offset_0;
        //int index_1 = global_id_1 - offset_1;
        int index = global_id_1 * global_size_0 + global_id_0;

        int f = global_id_0 + 100 * global_id_1;

        res_g[index] = f;
    }

    """


    prg = cl.Program(ctx, kernel).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, coordinate_set1.nbytes * coordinate_set2.nbytes)
    
    # Note that coordinate_set1 and coordinate_set2 shapes are swapped...
    #prg.id_check(queue, (coordinate_set2.shape[0], coordinate_set1.shape[0]), None, s_lat_g, s_lng_g, e_lat_g, e_lng_g, width_g, res_g)
    #res_np = np.zeros((coordinate_set1.shape[0]*coordinate_set2.shape[0],), dtype=np.int32)
    
    
    prg.cl_distance_array(queue, (coordinate_set2.shape[0], coordinate_set1.shape[0]), None, s_lat_g, s_lng_g, e_lat_g, e_lng_g, width_g, res_g)
    res_np = np.zeros((coordinate_set1.shape[0]*coordinate_set2.shape[0],), dtype=np.float32)

    
    cl.enqueue_copy(queue, res_np, res_g)

    # Do max_dist.
    _invalid_condition = (res_np<0)
    
    if (max_dist is not None and max_dist > 0):
        _invalid_condition = (_invalid_condition) | (res_np > max_dist)

    res_np = np.where(_invalid_condition, np.nan, res_np)

    
    return res_np.reshape(coordinate_set1.shape[0], coordinate_set2.shape[0])
    