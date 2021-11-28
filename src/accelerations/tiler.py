import sys
import math
from typing import Generator

import numpy as np
from accelerations.settings import DEFAULT_MEMORY_LIMIT, DEBUG

def dict_iter(obj:dict):
    yield zip(obj.keys(), obj.values())

def estimate_size(obj):
    if (isinstance(obj, np.ndarray)):
        return obj.nbytes
    
    if (isinstance(obj, (list, tuple)) and \
        not isinstance(obj, str)):
        return sum([estimate_size(_item) for _item in obj])
    elif (isinstance(obj, dict)):
        return sum([estimate_size(_item) for _item in obj.values()])
    else:
        return sys.getsizeof(obj)


# A tiler basically allows this:
#
####################################################
# mytiler = tiler(
#       inputs = whole_inputs,
#       outputs = whole_outputs,
#       memory_limit = memory_limit
# )
#
# for _tiled_inputs in mytiler.get_inputs():
#     _tiled_outputs = some_func(_tiled_inputs)
#     mytiler.put_outputs(_tiled_outputs)
####################################################
#
# ...which cuts the inputs into smaller tiles which is small enough to fit within memory_limit.
# After processing, it put the outputs in the correct location of the output tiles.
#
# This is mainly to keep GPU memory usage in check.

# <<< Do not use the superclass. It does nothing. >>>
class tiler():
    def __init__(
        self,
        inputs:dict = None,
        outputs:tuple = None,
        memory_limit:int = DEFAULT_MEMORY_LIMIT,
        show_progress:bool = DEBUG,
    )->None:
        self.reset_counter()

        self.show_progress = show_progress

        self.inputs = inputs if inputs else {}
        self.outputs = list(outputs) if outputs else []
        self.memory_limit = memory_limit

        self.memory_consumption = sum(
            (estimate_size(self.inputs),
            estimate_size(self.outputs))
        )

        self.min_tiles = self.memory_consumption/self.memory_limit
        self.max_tiles = np.prod([_arrInput.shape[0]/2 for _arrInput in self.get_array_inputs()])
        self.no_of_tiles = self.min_tiles
        self.tile_shape = (self.no_of_tiles,1)
    
    def skip_outputs(
        self,
        count:int=1,
    ):
        return self.set_counter(self.counter + count)

    def set_counter(
        self,
        count:int=0,
    ):
        self.counter = count
        return self.counter

    def reset_counter(
        self,
    ):
        return self.set_counter(0)
    
    def tiles(self):
        self.outputs = yield self.inputs
        yield None

    def get_array_inputs(self, limit:int=None):
        _inputs = []
        for _input in self.inputs.values():
            if (isinstance(_input, np.ndarray)):
                _inputs.append(_input)
                if (limit is not None):
                    if (len(_inputs) >= limit):
                        break

        return _inputs

class tiler_coordinates(tiler):
    def __init__(
        self,
        **kwargs,
    )->None:
        super().__init__(
            **kwargs,
        )

        self.array_inputs = self.get_array_inputs(2)

        if (kwargs.get("output", None) is None):
            self.outputs = (
                np.empty(
                    (self.array_inputs[0].shape[0],
                     self.array_inputs[1].shape[0],
                    ),
                    dtype=self.array_inputs[0].dtype,
                ),
            )

        # The simple min_tiles calculation in super() essentially calculates:
        #       (size_in1 + size_in2 + size_out)/(shape0*shape1) <= memory_limit
        # However the actual memory consumption is:
        #       (size_in1/shape0 +
        #        size_in2/shape1 + 
        #        size_out/(shape0*shape1))
        # which is strictly larger than what super() assumes.
        # Lets just use double the tiles or now.

        if (self.show_progress): print (f"Minimum number of tiles: {self.min_tiles}")

        _optimised = False
        while (not(_optimised)):

            self.calculate_tile_shape()

            self.no_of_tiles = self.tile_shape[0] * self.tile_shape[1]

            if (self.show_progress): print(f"Proposed Tile shape: {self.tile_shape}")

            if (self.show_progress): print (f"Proposed number of tiles: {self.no_of_tiles}")
            if (self.show_progress): print (self.max_tiles)

            _tile_memory_consumption = self.calculate_tile_memory_consumption()
            if (self.show_progress): print (f"Proposed memory consumption per tile: {_tile_memory_consumption}")

            if (not(_tile_memory_consumption <= self.memory_limit) and \
                self.min_tiles < self.max_tiles):
                self.min_tiles = max(
                    self.no_of_tiles * (_tile_memory_consumption/self.memory_limit),
                    self.min_tiles+1
                    )
                if (self.show_progress): print (f"Minimum number of tiles: {self.min_tiles}")
            else:
                _optimised = True
                break

    def calculate_tile_memory_consumption(self)->int:
        return sum(
            (
                estimate_size(self.array_inputs[0])/self.tile_shape[0],
                estimate_size(self.array_inputs[1])/self.tile_shape[1],
                estimate_size(self.outputs[0])/self.no_of_tiles,
            )
        )

    def calculate_tile_size(self):
        self.tile_size = (
            math.ceil(self.array_inputs[0].shape[0] / self.tile_shape[0]),
            math.ceil(self.array_inputs[1].shape[0] / self.tile_shape[1]),
        )

    def calculate_tile_shape(self):

        _smaller_input  =   int(self.array_inputs[0].shape[0] >= self.array_inputs[1].shape[0])
        _larger_input   =   int(not _smaller_input)

        # First propose a tile shape
        self.tile_shape = [None,None]

        self.tile_shape[_smaller_input] =  math.ceil((
            self.array_inputs[_smaller_input].shape[0] * self.min_tiles / self.array_inputs[_larger_input].shape[0]
        ) ** 0.5)

        self.tile_shape[_larger_input]  =   math.ceil(self.min_tiles/self.tile_shape[_smaller_input])

        # Then we calculate the tile size
        self.calculate_tile_size()

        # This is the funny bit
        # Assume you have 10 rows in array_input[0], and tile_shape[0] is 6,
        # you will end up with a tile_size[0] of 2.
        # However in order to consume 10 rows of size 2, we only need tile_shape[1] == 5.
        # This is to readjust the tile_shape after we have the tile_size.
        self.tile_shape = (
            math.ceil(self.array_inputs[0].shape[0] / self.tile_size[0]),
            math.ceil(self.array_inputs[1].shape[0] / self.tile_size[1]),
        )
        

    def tiles(self)->Generator:
        _no_of_array_inputs = 2

        for _tile in range(self.no_of_tiles):

            if (self.show_progress): print ("="*60)
            if (self.show_progress): print (f"TILE No. {_tile}")

            _tile_xn = _tile // self.tile_shape[1]
            _tile_yn = _tile % self.tile_shape[1]

            _tile_xrange = [
                _tile_xn * self.tile_size[0],
                min((_tile_xn+1) * self.tile_size[0], self.array_inputs[0].shape[0]),
            ]

            _tile_yrange = [
                _tile_yn * self.tile_size[1],
                min((_tile_yn+1) * self.tile_size[1], self.array_inputs[1].shape[0]),
            ]

            _tile_size = (
                _tile_xrange[1]-_tile_xrange[0],
                _tile_yrange[1]-_tile_yrange[0],
            )
            
            # This is not needed strictly speaking - just a record
            self.set_counter(_tile)

            # Copy over the array inputs
            _tiled_inputs = self.inputs.copy()
            
            # Replace the array inputs
            for _key, _input in dict_iter(_tiled_inputs):
                if (_input is self.array_inputs[0]):
                    _tiled_inputs[_key] = self.array_inputs[0][_tile_xrange[0]:_tile_xrange[1],]
                elif (_input is self.array_inputs[1]):
                    _tiled_inputs[_key] = self.array_inputs[1][_tile_yrange[0]:_tile_yrange[1],]

            # Push inputs out and wait for tiler.send(_tiled_outputs)
            _tiled_outputs = (yield _tiled_inputs)

            if (not isinstance(_tiled_outputs, tuple)):
                _tiled_outputs = (_tiled_outputs, )

            if (self.show_progress): print (sum(
                (
                    _tiled_inputs["input1"].nbytes,
                    _tiled_inputs["input2"].nbytes,
                    _tiled_outputs[0].nbytes
                ),
            ),
            self.memory_limit,)

            for _id, _output in enumerate(_tiled_outputs):

                assert _output.shape == _tile_size
                
                self.outputs[_id][
                    _tile_xrange[0]:_tile_xrange[1],
                    _tile_yrange[0]:_tile_yrange[1],
                ] = _output

        yield None

class tiler_byte_operations(tiler):
    def __init__(
        self,
        **kwargs,
    )->None:
        super().__init__(
            **kwargs,
        )

    def tiles(self):
        yield self.inputs
        self.outputs = yield

class tiler_hashing(tiler):
    def __init__(
        self,
        inputs:dict = None,
        outputs:tuple = None,
        memory_limit:int = DEFAULT_MEMORY_LIMIT,
    )->None:
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            memory_limit=memory_limit,
        )

    def tiles(self):
        yield self.inputs
        self.outputs = yield
        




# if __name__ == "__main__":

#     _n1 = 1200
#     _n2 = 4400
#     _arrInput1 = np.arange(_n1, dtype=np.float64).reshape((_n1//4,4))
#     _arrInput2 = np.arange(_n2, dtype=np.float64).reshape((_n2//4,4))

#     print (f"Shape of Input 1: {_arrInput1.shape}")
#     print (f"Shape of Input 2: {_arrInput2.shape}")

#     # _arrOutput = np.empty((_arrInput1.shape[0], _arrInput2.shape[0]), dtype=np.uint64)

#     def dot(input1, input2):
#         return np.dot(input1, input2.T)

#     print ("ANSWER")
#     print (_answer := dot(_arrInput1, _arrInput2))

#     print()

#     _tiler = tiler_coordinates(
#         inputs={
#             "input1":_arrInput1,
#             "input2":_arrInput2,
#         },
#         memory_limit=4096,
#     )

#     print (f"To be processed with tile shape {_tiler.tile_shape}.")
#     _arrOutput = _tiler.outputs[0]

#     _tiler_gen = _tiler.tiles()

#     _tiled_input = next(_tiler_gen)
#     while (_tiled_input := _tiler_gen.send((dot(**_tiled_input),))) is not None:
#         pass
    
#     print (_arrOutput)
#     np.testing.assert_array_equal(_answer, _arrOutput)