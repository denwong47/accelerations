import math
from re import M

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
                                      AcceleratedProcessInvalidInput
from accelerations.tiler import tiler_coordinates
from accelerations.settings import DEFAULT_MEMORY_LIMIT, CUDA_DEFAULT_BLOCK_DIM
from accelerations.settings import DEBUG, DEBUG_TILER

# Extracted from https://www.movable-type.co.uk/scripts/geodesy/docs/module-latlon-ellipsoidal-LatLonEllipsoidal.html
ELLIPSE_WGS84_A = 6378.137          # km
ELLIPSE_WGS84_B = 6356.752314245    # km
ELLIPSE_WGS84_F = 1/298.257223563

VINCENTY_MAX_ITERATIONS = 1000

HAVERSINE_RADIUS = 6373.0

EPS = 2**-52
# ========================================================================================

def geodistance_sphr_between_two_latlngs(
    s_lat:np.float64,
    s_lng:np.float64,
    e_lat:np.float64,
    e_lng:np.float64,
    )->np.float64:
    # Haversine calculation
    # Assumes spherical world - fast but has errors up to ~0.35%
    # Copied from https://towardsdatascience.com/better-parallelization-with-numba-3a41ca69452e
    
    # approximate radius of earth in km
    R = HAVERSINE_RADIUS
    
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


njit_geodistance_sphr_between_two_latlngs   = numba.njit(geodistance_sphr_between_two_latlngs)
# ========================================================================================

def geodistance_ellip_between_two_latlngs(
    s_lat:np.float64,
    s_lng:np.float64,
    e_lat:np.float64,
    e_lng:np.float64,
    )->np.float64:
    # Vincenty solutions of geodescis on the ellipsoid
    # Adapted from https://www.movable-type.co.uk/scripts/latlong-vincenty.html
    # https://github.com/mrJean1/PyGeodesy

    # Geodesics on Ellipsoid calculation - accurate but iterative

    # Discard calculation if latitude doesn't make any sense
    # This algorithm does not work when Latitude = 90 or -90

    if (-90 < s_lat < 90 and -90 < e_lat < 90):

        if (abs(s_lat-e_lat) <= EPS and \
            abs(s_lng-e_lng) <= EPS):
            # Coincidence will cause a TypeError below - escape
            return 0
    
        s_lat = s_lat * math.pi / 180                     
        s_lng = s_lng * math.pi / 180 
        e_lat = e_lat * math.pi / 180                    
        e_lng = e_lng * math.pi / 180

        diff_lng = e_lng - s_lng # L = difference in longitude, U = reduced latitude, defined by tan U = (1-f)·tanφ.
        tan_reduced_s_lat = (1-ELLIPSE_WGS84_F) * math.tan(s_lat)
        cos_reduced_s_lat = 1 / math.sqrt((1 + tan_reduced_s_lat*tan_reduced_s_lat))
        sin_reduced_s_lat = tan_reduced_s_lat * cos_reduced_s_lat

        tan_reduced_e_lat = (1-ELLIPSE_WGS84_F) * math.tan(e_lat)
        cos_reduced_e_lat = 1 / math.sqrt((1 + tan_reduced_e_lat*tan_reduced_e_lat))
        sin_reduced_e_lat = tan_reduced_e_lat * cos_reduced_e_lat

        _lambda = diff_lng
        sin_lng = None
        cos_lng = None    # λ = difference in longitude on an auxiliary sphere

        ang_dist = None
        sin_ang_dist = None
        cos_ang_dist = None # σ = angular distance P₁ P₂ on the sphere

        cos_2_ang_dist_from_equator_bisect = None                      # σ' = angular distance on the sphere from the equator to the midpoint of the line
        cos_sq_azimuth_of_geodesic_at_equator = None                   # α = azimuth of the geodesic at the equator

        _lamdba_dash = None

        for _ in range(VINCENTY_MAX_ITERATIONS):
            sin_lng = math.sin(_lambda)
            cos_lng = math.cos(_lambda)

            sin_sq_ang_dist =   (cos_reduced_e_lat*sin_lng) * (cos_reduced_e_lat*sin_lng) + \
                                (cos_reduced_s_lat*sin_reduced_e_lat-sin_reduced_s_lat*cos_reduced_e_lat*cos_lng)**2

            # if (abs(sin_sq_ang_dist) < EPS):
            #     break

            sin_ang_dist = math.sqrt(sin_sq_ang_dist)
            cos_ang_dist = sin_reduced_s_lat*sin_reduced_e_lat + cos_reduced_s_lat*cos_reduced_e_lat*cos_lng
            
            ang_dist = math.atan2(sin_ang_dist, cos_ang_dist)

            sin_azimuth_of_geodesic_at_equator = cos_reduced_s_lat * cos_reduced_e_lat * sin_lng / sin_ang_dist
            cos_sq_azimuth_of_geodesic_at_equator = 1 - sin_azimuth_of_geodesic_at_equator**2

            if (abs(cos_sq_azimuth_of_geodesic_at_equator) > EPS):
                cos_2_ang_dist_from_equator_bisect = cos_ang_dist - 2*sin_reduced_s_lat*sin_reduced_e_lat/cos_sq_azimuth_of_geodesic_at_equator
            else:
                cos_2_ang_dist_from_equator_bisect = 0  # on equatorial line cos²α = 0

            _C = ELLIPSE_WGS84_F/16*cos_sq_azimuth_of_geodesic_at_equator*(4+ELLIPSE_WGS84_F*(4-3*cos_sq_azimuth_of_geodesic_at_equator))

            _lamdba_dash = _lambda
            _lambda =   diff_lng + (1-_C) * ELLIPSE_WGS84_F * sin_azimuth_of_geodesic_at_equator * \
                        (ang_dist + _C*sin_ang_dist*(cos_2_ang_dist_from_equator_bisect+_C*cos_ang_dist*(-1+2*cos_2_ang_dist_from_equator_bisect*cos_2_ang_dist_from_equator_bisect)))

            if (abs(_lambda-_lamdba_dash) <= 1e-12):
                break

        _uSq = cos_sq_azimuth_of_geodesic_at_equator * (ELLIPSE_WGS84_A**2 - ELLIPSE_WGS84_B**2) / (ELLIPSE_WGS84_B**2)
        _A = 1 + _uSq/16384*(4096+_uSq*(-768+_uSq*(320-175*_uSq)))
        _B = _uSq/1024 * (256+_uSq*(-128+_uSq*(74-47*_uSq)))
        delta_ang_dist =    _B*sin_ang_dist*(cos_2_ang_dist_from_equator_bisect+ \
                            _B/4*(cos_ang_dist*(-1+2*cos_2_ang_dist_from_equator_bisect*cos_2_ang_dist_from_equator_bisect)-\
                            _B/6*cos_2_ang_dist_from_equator_bisect*(-3+4*sin_ang_dist*sin_ang_dist)*(-3+4*cos_2_ang_dist_from_equator_bisect*cos_2_ang_dist_from_equator_bisect)))

        return ELLIPSE_WGS84_B*_A*(ang_dist-delta_ang_dist) # s = length of the geodesic

    else:
        # Returning NaN to cuda_distance_array means irrelevance
        return njit_geodistance_sphr_between_two_latlngs(s_lat, s_lng, e_lat, e_lng)


njit_geodistance_ellip_between_two_latlngs  = numba.njit(geodistance_ellip_between_two_latlngs)


# ========================================================================================

cuda_geodistance_sphr_between_two_latlngs   = cuda.jit(device=True)(geodistance_sphr_between_two_latlngs)
cuda_geodistance_ellip_between_two_latlngs  = cuda.jit(device=True)(geodistance_ellip_between_two_latlngs)

@cuda.jit
def cuda_geodistance_between_arrays(
    input1:np.ndarray,
    input2:np.ndarray,
    max_dist:np.float64,
    precise:np.bool8,
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
            if (precise):
                dist = cuda_geodistance_ellip_between_two_latlngs(input1[x,0], input1[x,1],
                                                                  input2[y,0], input2[y,1])
            else:
                dist = cuda_geodistance_sphr_between_two_latlngs(input1[x,0], input1[x,1],
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
    

class geodistance(accelerated_process):

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
            return AcceleratedProcessInvalidInput("geodistance require both input1 and input2. All input parameters needed to be keyworded.")
    
    def process_cpu(
            input1:np.ndarray,
            input2:np.ndarray,
            max_dist:np.float64=-1,
            precise:bool=False,
            dtype:type=np.double,
            memory_limit:int=DEFAULT_MEMORY_LIMIT,
        )->np.ndarray:
        output = np.empty(
            shape=(
                input1.shape[0],
                input2.shape[0],
            ),
            dtype=dtype,
        )

        for _point1 in numba.prange(input1.shape[0]):
            for _point2 in numba.prange(input2.shape[0]):
                s_lat = input1[_point1, 0]
                s_lng = input1[_point1, 1]
                e_lat = input2[_point2, 0]
                e_lng = input2[_point2, 1]

                if (math.fabs(s_lat - e_lat) <= max_dist/100 or max_dist < 0):
                    if (precise):
                        output[_point1, _point2]   =   njit_geodistance_ellip_between_two_latlngs(
                            s_lat = s_lat,
                            s_lng = s_lng,
                            e_lat = e_lat,
                            e_lng = e_lng,
                        )
                    else:
                        output[_point1, _point2]   =   njit_geodistance_sphr_between_two_latlngs(
                            s_lat = s_lat,
                            s_lng = s_lng,
                            e_lat = e_lat,
                            e_lng = e_lng,
                        )
                else:
                    output[_point1, _point2]       =   np.nan

        return output
    
    # For Parallel, we just need to @numba.njit it
    process_cpu_parallel = numba.njit(process_cpu, parallel=True)
    # process_cpu_parallel = accelerated_process.tile_process(
    #     tiler_class=tiler_coordinates,
    #     show_progress=DEBUG_TILER, #use DEBUG if you want to debug tiler
    #     )(process_cpu_parallel)
    process_cpu = numba.njit(process_cpu)

    @accelerated_process.tile_process(
        tiler_class=tiler_coordinates,
        show_progress=DEBUG_TILER, #use DEBUG if you want to debug tiler
        )
    def process_cuda(
        input1:np.ndarray,  # Do not rename - it has to be called input\d
        input2:np.ndarray,  # Do not rename - it has to be called input\d
        max_dist:np.float64=-1,
        precise:bool=False,
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

        _no_of_dimensions = 2

        output = np.empty((input1.shape[0], input2.shape[0]), dtype=dtype)

        _ondevice_input1 = cuda.to_device(np.ascontiguousarray(input1[:,:_no_of_dimensions]))
        _ondevice_input2 = cuda.to_device(np.ascontiguousarray(input2[:,:_no_of_dimensions]))
        _ondevice_output = cuda.to_device(output)
        
        if (show_progress):
            print ("DATA   DIM : (%d, %d)" % (input1.shape[0], input2.shape[0]))
            print (" : ".join(["GRID   DIM", str(grid_dim)]))
            print (" : ".join(["BLOCK  DIM", str(block_dim)]))

        cuda_geodistance_between_arrays[grid_dim, block_dim](
            _ondevice_input1,
            _ondevice_input2,
            max_dist,
            precise,
            _ondevice_output
        )

        _ondevice_output.copy_to_host(output)

        return output

    process_opencl = process_cpu_parallel
    

# if (__name__=="__main__"):
    # # Generate random arrays
    # _counts = (997, 127)
    # _rng = np.random.default_rng()
    # _coors = [np.array(
    #     [ (_x, _y) for _x,_y in zip(_rng.uniform(-90,90,size=_count), _rng.uniform(-180,180,size=_count)) ],
    #     dtype=np.float64
    # ) for _count in _counts]

    # np.set_printoptions(precision=3, suppress=True, linewidth=200)

    # from execute_timer import execute_timer
    
    # with execute_timer(echo=True) as _timer:
    #     _cuda = geodistance.process_cuda(
    #         input1=_coors[0],
    #         input2=_coors[1],
    #         max_dist=-1,
    #         precise=False,
    #         show_progress=True
    #     )

    # with execute_timer(echo=True) as _timer:
    #     _haversine = geodistance.process_cpu_parallel(
    #         input1=_coors[0],
    #         input2=_coors[1],
    #         max_dist=-1,
    #         precise=True
    #     )

    # with execute_timer(echo=True) as _timer:
    #     _vincenty = geodistance.process_cpu_parallel(
    #         input1=_coors[0],
    #         input2=_coors[1],
    #         max_dist=-1,
    #         precise=False
    #     )
    
    # # with execute_timer(echo=True) as _timer:
    # #     _single = geodistance.process_cpu(
    # #         _coors,
    # #         _coors,
    # #         -1,
    # #         precise=True
    # #     )
        
    # # print (np.abs(_cuda-_parallel)/_parallel*100)