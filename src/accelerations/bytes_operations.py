import enum
import math
from typing import Any, Iterable, Union

from numba.np.ufunc import parallel
import numpy as np
import numba
from numba import cuda

from accelerations.accelerator import accelerated_process, \
                                      accelerator_type, \
                                      AcceleratedProcessInvalidInput
from accelerations.tiler import tiler_byte_operations

from accelerations.settings import DEFAULT_MEMORY_LIMIT, CUDA_DEFAULT_BLOCK_DIM, CUDA_DEFAULT_THREAD_PER_BLOCK
from accelerations.settings import DEBUG



class ubyteArrayRequired(ValueError):
    def __bool__(self):
        return False
    __nonzero__=__bool__

def bytes_to_np(data:bytes) -> np.ndarray:
    if (isinstance(data, np.ndarray)):
        return data
    elif (isinstance(data, bytes)):
        return np.frombuffer(data, dtype=np.uint8)
    else:
        return bytes_to_np(repr(data).encode("utf-8"))

def np_to_bytes(ndarray:np.ndarray) -> bytes:
    if (isinstance(ndarray, bytes)):
        return ndarray
    elif (isinstance(ndarray, np.ndarray)):
        return ndarray.tobytes()
    else:
        raise ubyteArrayRequired(f"np.ndarray of dtype uint8 expected, {type(ndarray)} found.")

def bits_unpacked(data:bytes):
    if (not(isinstance(data, np.ndarray))):
        data = bytes_to_np(data)

    return np.unpackbits(data)

def bits_packed(
    bits_array:np.ndarray,
    kind:type=np.ndarray
    ) -> np.ndarray:
    
    _return = np.packbits(bits_array.astype(dtype=np.uint8))

    if (_return is np.ndarray):
        return _return
    elif (_return is bytes):
        return np_to_bytes(_return)


def pad_array_to_length(
    array:np.ndarray,
    length:int,
) -> np.ndarray:

    _rng = np.random.default_rng(array)
    return _rng.integers(0, 255, length, dtype=array.dtype)    

array_binary_repr = lambda a: np.vectorize(np.binary_repr)(a, width=a.dtype.alignment*8)

def cast_int_sequentially_with_specified_length(
    input1:np.ndarray,
    dtype:Union[
        np.dtype,
        str,
    ],
    input_length:int,
    output_length:int,
):
    """
    Cast integer arrays into a dtype of different length, sequentially.

    1D array only.

    For example if you have an np.arange(8, dtype=np.uint8):
    uint8 ['00000000' '00000001' '00000010' '00000011' '00000100' '00000101' '00000110' '00000111']

    You can cast them into np.uint16:
    uint16 ['00000000 00000001'  '00000010 00000011'   '00000100 00000101'   '00000110 00000111'  ]
    or np.uint32:
    uint32 ['00000000 00000001 00000010 00000011'      '00000100 00000101 00000110 00000111']

    Note that this is different from np.frombuffer or np.view - those would have reversed the order of each byte when grouping:
    e.g.
    uint32 ['00000011 00000010 00000001 00000000'      '00000111 00000110 00000101 00000100']

    The principle use of this function is for hashing - when casting 4 bytes of data into a word of 32 bits.


    There are more elegant ways to write this function - but we need to maintain CUDA compatibility,
    so np.apply() etc cannot be used.
    """
    
    _input_data_size    =   input_length
    _output_data_size   =   output_length

    _cast_ratio         =   _output_data_size/_input_data_size

    output1 = np.empty(
        (math.ceil(input1.shape[0]/_cast_ratio),),
        dtype=dtype,
    )


    for _output_pos in numba.prange(output1.shape[0]):
        _input_pos  =   math.floor(_output_pos * _cast_ratio)

        if (_cast_ratio > 1):
            """
            Casting from shorter to longer dtypes
            """
            _cast_ratio = int(_cast_ratio)

            output1[_output_pos]    =   np.sum(
                input1[_input_pos:_input_pos+_cast_ratio].astype(dtype=dtype) << (np.arange(_cast_ratio-1, -1, -1, dtype=np.uint8)*8),
            )
        elif (_cast_ratio < 1):
            """
            Casting from longer to shorter dtypes
            """
            _input_offset           =   int(1/_cast_ratio - _output_pos % (1/_cast_ratio) - 1)
            output1[_output_pos]    =   input1[_input_pos] >> (_input_offset * 8)       # we should not need to remove the overflowing values if the output dtype is correctly set
        else:
            """
            What are we casting????
            """
            return input1.astype(dtype=dtype)

    return output1

njit_cast_int_sequentially_with_specified_length = numba.njit(parallel=True)(cast_int_sequentially_with_specified_length)
cuda_cast_int_sequentially_with_specified_length = cuda.jit(device=True)(cast_int_sequentially_with_specified_length)

def cast_int_sequentially(
    input1:np.ndarray,
    dtype:Union[
        np.dtype,
        str,
    ],
):
    _input_data_size    =   input1.dtype.alignment* 8
    _output_data_size   =   np.dtype(dtype).alignment * 8

    return cast_int_sequentially_with_specified_length(
        input1=input1,
        dtype=dtype,
        input_length=_input_data_size,
        output_length=_output_data_size,
    )

def njit_cast_int_sequentially(
    input1:np.ndarray,
    dtype:Union[
        np.dtype,
        str,
    ],
):
    _input_data_size    =   input1.dtype.alignment* 8
    _output_data_size   =   np.dtype(dtype).alignment * 8

    return njit_cast_int_sequentially_with_specified_length(
        input1=input1,
        dtype=dtype,
        input_length=_input_data_size,
        output_length=_output_data_size,
    )

# ========================================================================================

def bytes_arrays_xor(
        bytes1:np.ndarray,
        bytes2:np.ndarray,
    )->np.ndarray:
    _repeats = math.ceil(bytes1.shape[0]/bytes2.shape[0])
    return np.bitwise_xor(
        bytes1,
        #np.tile(bytes2, math.ceil(bytes1.shape[0]/bytes2.shape[0]))[:bytes1.shape[0]],
        bytes2.repeat(_repeats).reshape((-1,_repeats)).T.reshape(-1)[:bytes1.shape[0]]
    )
# ========================================================================================

@numba.njit(parallel=True)
def njit_bytes_arrays_xor(
        bytes1:np.ndarray,
        bytes2:np.ndarray,
    ):
    bytes_return = np.empty_like(bytes1)

    for _pos in numba.prange(bytes1.shape[0]):
        bytes_return[_pos] = bytes1[_pos] ^ bytes2[_pos % bytes2.shape[0]]
    
    return bytes_return


# ========================================================================================

# There is really no reason to use CUDA in this module - consider using CPU_MULTIPROCESS instead.
@cuda.jit
def cuda_bytes_arrays_xor(
    bytes1:np.ndarray,
    bytes2:np.ndarray,
    bytes_return:np.ndarray,
):
    # When running a kernel, the kernel function’s code is executed by every thread once.
    # It therefore has to know which thread it is in, in order to know which array element(s) it is responsible for (complex algorithms may define more complex responsibilities, but the underlying principle is the same).
    x, y = cuda.grid(2)
    
    # The above is a shorthand for below:
    # This ignores all the complexity of trying to work out the location of the block etc.
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    if x < bytes1.shape[0]:
        bytes_return[x] = bytes1[x] ^ bytes2[x % bytes2.shape[0]]
    else:
        return

# ========================================================================================

def _bytes_arrays_shift_with_specified_length(
        bytes1:np.ndarray,
        amount:int,
        rotate:bool=False,
        length:int=None, # njit does not support alignment - this needs to be set everytime
    )->np.ndarray:
    
    bytes_return = np.empty_like(bytes1)

    np_int = lambda num: np.full((1), num, dtype=bytes1.dtype)[0] # Convert int to relevant np dtype. This is necessary for left_shift and right_shift to work.
    
    if (rotate):
        _mask_right = np_int(np.sum(2**np.arange(abs(amount), dtype=bytes1.dtype)))
        _mask_left = _mask_right << np_int(length - abs(amount))

    for _pos in numba.prange(bytes1.shape[0]):

        # _origin = bytes1[_pos]

        if (amount > 0):
            bytes_return[_pos] = bytes1[_pos] << np_int(amount)
        else:
            bytes_return[_pos] = bytes1[_pos] >> np_int(-amount)

        # _shift = bytes_return[_pos]
            
        if (rotate):
            if (amount > 0):
                # _mask = _mask_left & bytes1[_pos]
                bytes_return[_pos] += (_mask_left & bytes1[_pos]) >> np_int(length - amount)
            else:
                # _mask = _mask_right & bytes1[_pos]
                bytes_return[_pos] += (_mask_right & bytes1[_pos]) << np_int(length + amount)
        else:
            pass
            # _mask = 0

        # _rotate = bytes_return[_pos]
        # print (f"{_pos:8}{' '*22}{bin(_origin):30}{bin(_shift):30}{bin(_mask):30}{bin(_rotate):30}")

    return bytes_return


njit_bytes_arrays_shift_with_specified_length = numba.njit(parallel=True)(_bytes_arrays_shift_with_specified_length)

# ========================================================================================

def bytes_arrays_shift(
        bytes1:np.ndarray,
        amount:int,
        rotate:bool=False,
    )->np.ndarray:

    return _bytes_arrays_shift_with_specified_length(
        bytes1=bytes1,
        amount=amount,
        rotate=rotate,
        length=bytes1.dtype.alignment*8, # njit does not support alignment - we have to do this outside
    )

# ========================================================================================

def njit_bytes_arrays_shift(
        bytes1:np.ndarray,
        amount:int,
        rotate:bool=False,
    )->np.ndarray:

    return njit_bytes_arrays_shift_with_specified_length(
        bytes1=bytes1,
        amount=amount,
        rotate=rotate,
        length=bytes1.dtype.alignment*8, # njit does not support alignment - we have to do this outside
    )

# ========================================================================================

@cuda.jit
def cuda_bytes_arrays_shift(
        bytes1:np.ndarray,
        abs_amount:int,
        direction:bool,
        length:int,
        mask_left:int,
        mask_right:int,
        bytes_return:np.ndarray,
    )->np.ndarray:

    # When running a kernel, the kernel function’s code is executed by every thread once.
    # It therefore has to know which thread it is in, in order to know which array element(s) it is responsible for (complex algorithms may define more complex responsibilities, but the underlying principle is the same).
    x, y = cuda.grid(2)
    
    # The above is a shorthand for below:
    # This ignores all the complexity of trying to work out the location of the block etc.
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    if x < bytes1.shape[0]:
        if (direction):
            bytes_return[x] = bytes1[x] << abs_amount
            bytes_return[x] += (mask_left & bytes1[x]) >> (length - abs_amount)
        else:
            bytes_return[x] = bytes1[x] >> abs_amount
            bytes_return[x] += (mask_right & bytes1[x]) << (length - abs_amount)            
    else:
        return

# ========================================================================================

# class bytes_operations_type(enum.Enum):
#     XOR = enum.auto()
#     ROL = enum.auto()
#     ROR = enum.auto()
#     SHL = enum.auto()
#     SHR = enum.auto()

class bytes_operations(accelerated_process):
   
    @staticmethod
    def byte_shift_cuda(
        bytes1:np.ndarray,
        amount:int,
        rotate:bool,
        block_dim:tuple=(CUDA_DEFAULT_THREAD_PER_BLOCK, 1),
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        show_progress:bool=False,
        **kwargs,
    )->np.ndarray:
        grid_dim = (
            math.ceil(bytes1.shape[0]/CUDA_DEFAULT_THREAD_PER_BLOCK),
            1
        )
        
        output = np.empty_like(bytes1)

        np_int = lambda num: np.full((1), num, dtype=bytes1.dtype)[0] # Convert int to relevant np dtype. This is necessary for left_shift and right_shift to work.

        abs_amount = np_int(abs(amount))
        direction = (amount>0)
        length = np_int(bytes1.dtype.alignment*8)
        
        if (rotate):
            _mask_right = np_int(np.sum(2**np.arange(abs_amount, dtype=bytes1.dtype)))
            _mask_left = _mask_right << np_int(length - abs_amount)
        else:
            _mask_right = 0
            _mask_left = 0

        _ondevice_input1 = cuda.to_device(np.ascontiguousarray(bytes1))
        _ondevice_output = cuda.to_device(output)

        if (show_progress):
            print ("DATA   DIM : (%d, %d)" % bytes1.shape)
            print (" : ".join(["GRID   DIM", str(grid_dim)]))
            print (" : ".join(["BLOCK  DIM", str(block_dim)]))

        cuda_bytes_arrays_shift[grid_dim, block_dim](
            _ondevice_input1,
            abs_amount,
            direction,
            length,
            _mask_left,
            _mask_right,
            _ondevice_output,
        )

        _ondevice_output.copy_to_host(output)

        return output


class bytes_XOR(bytes_operations):
    process_cpu = bytes_arrays_xor
    process_cpu_parallel = njit_bytes_arrays_xor
    
    def process_cuda(
        bytes1:np.ndarray,
        bytes2:np.ndarray,
        block_dim:tuple=(CUDA_DEFAULT_THREAD_PER_BLOCK, 1),
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        show_progress:bool=False,
        **kwargs,
    )->np.ndarray:
        grid_dim = (
            math.ceil(bytes1.shape[0]/CUDA_DEFAULT_THREAD_PER_BLOCK),
            1
        )
        
        output = np.empty_like(bytes1)

        _ondevice_input1 = cuda.to_device(np.ascontiguousarray(bytes1))
        _ondevice_input2 = cuda.to_device(np.ascontiguousarray(bytes2))
        _ondevice_output = cuda.to_device(output)

        if (show_progress):
            print ("DATA   DIM : (%d, %d)" % bytes1.shape)
            print (" : ".join(["GRID   DIM", str(grid_dim)]))
            print (" : ".join(["BLOCK  DIM", str(block_dim)]))

        cuda_bytes_arrays_xor[grid_dim, block_dim](
            _ondevice_input1,
            _ondevice_input2,
            _ondevice_output
        )

        _ondevice_output.copy_to_host(output)

        return output

    process_opencl = process_cpu_parallel

class bytes_ROL(bytes_operations):

    def process_cpu(
        bytes1:np.ndarray,
        amount:int,
    ):
        return bytes_arrays_shift(
            bytes1,
            amount,
            rotate=True,
        )

    def process_cpu_parallel(
        bytes1:np.ndarray,
        amount:int,
    ):
        return njit_bytes_arrays_shift(
            bytes1,
            amount,
            rotate=True,
        )
  
    def process_cuda(
        bytes1:np.ndarray,
        amount:int,
        block_dim:tuple=(CUDA_DEFAULT_THREAD_PER_BLOCK, 1),
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        show_progress:bool=False,
        **kwargs,
    ):
        return bytes_operations.byte_shift_cuda(
            bytes1,
            amount,
            rotate=True,
            block_dim=block_dim,
            memory_limit=memory_limit,
            show_progress=show_progress,
            **kwargs,
        )

class bytes_ROR(bytes_operations):

    def process_cpu(
        bytes1:np.ndarray,
        amount:int,
    ):
        return bytes_arrays_shift(
            bytes1,
            -amount,
            rotate=True,
        )

    def process_cpu_parallel(
        bytes1:np.ndarray,
        amount:int,
    ):
        return njit_bytes_arrays_shift(
            bytes1,
            -amount,
            rotate=True,
        )

    def process_cuda(
        bytes1:np.ndarray,
        amount:int,
        block_dim:tuple=(CUDA_DEFAULT_THREAD_PER_BLOCK, 1),
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        show_progress:bool=False,
        **kwargs,
    ):
        return bytes_operations.byte_shift_cuda(
            bytes1,
            -amount,
            rotate=True,
            block_dim=block_dim,
            memory_limit=memory_limit,
            show_progress=show_progress,
            **kwargs,
        )

class bytes_SHL(bytes_operations):

    def process_cpu(
        bytes1:np.ndarray,
        amount:int,
    ):
        return bytes_arrays_shift(
            bytes1,
            amount,
            rotate=False,
        )

    def process_cpu_parallel(
        bytes1:np.ndarray,
        amount:int,
    ):
        return njit_bytes_arrays_shift(
            bytes1,
            amount,
            rotate=False,
        )

    def process_cuda(
        bytes1:np.ndarray,
        amount:int,
        block_dim:tuple=(CUDA_DEFAULT_THREAD_PER_BLOCK, 1),
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        show_progress:bool=False,
        **kwargs,
    ):
        return bytes_operations.byte_shift_cuda(
            bytes1,
            amount,
            rotate=False,
            block_dim=block_dim,
            memory_limit=memory_limit,
            show_progress=show_progress,
            **kwargs,
        )

class bytes_SHR(bytes_operations):

    def process_cpu(
        bytes1:np.ndarray,
        amount:int,
    ):
        return bytes_arrays_shift(
            bytes1,
            -amount,
            rotate=False,
        )

    def process_cpu_parallel(
        bytes1:np.ndarray,
        amount:int,
    ):
        return njit_bytes_arrays_shift(
            bytes1,
            -amount,
            rotate=False,
        )

    def process_cuda(
        bytes1:np.ndarray,
        amount:int,
        block_dim:tuple=(CUDA_DEFAULT_THREAD_PER_BLOCK, 1),
        memory_limit:int=DEFAULT_MEMORY_LIMIT,
        show_progress:bool=False,
        **kwargs,
    ):
        return bytes_operations.byte_shift_cuda(
            bytes1,
            -amount,
            rotate=False,
            block_dim=block_dim,
            memory_limit=memory_limit,
            show_progress=show_progress,
            **kwargs,
        )


if __name__=="__main__":
    # for _device_type in accelerator_type:
    #     _process = bytes_XOR.process(type=_device_type)

    #     _args = {
    #         'bytes1': np.arange(100,
    #                     dtype=np.uint8),
    #         'bytes2': np.array([1, 0],
    #                     dtype=np.uint8)}

    #     print(bytes_XOR.process_cpu_parallel(**_args))

    _bytes1 = np.random.randint(0, 2**32, 2**20, dtype=np.uint32)

    _amount = 26
    _rotate = True

    _bytesNJIT = bytes_ROR.process_cpu_parallel(_bytes1,_amount)
    _bytesCUDA = bytes_ROR.process_cuda(_bytes1,_amount)

    print (_bytesNJIT)
    print (_bytesCUDA)
    print (np.testing.assert_equal(_bytesNJIT, _bytesCUDA))