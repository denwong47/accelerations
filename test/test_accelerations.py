from ast import Import
import os
import unittest

import logging
import pickle

import inspect
import numpy as np
from accelerations.accelerator import accelerator_type
from accelerations.multi_dimensional_distance import multi_dimensional_distance, distance_between_two_points
from accelerations.geodistance import njit_geodistance_ellip_between_two_latlngs, njit_geodistance_sphr_between_two_latlngs, geodistance
from accelerations.bytes_operations import bytes_XOR, bytes_operations, bytes_arrays_xor, bytes_to_np, np_to_bytes, pad_array_with_random, cuda_bytes_arrays_xor

try:
    import numba
    from numba import cuda

    if (not cuda.detect()):
        cuda = None
except (ImportError, ModuleNotFoundError) as e:
    numba = None
    cuda = None

from file_io import file
from execute_timer import execute_timer

logging.basicConfig(filename="test.log", level=logging.DEBUG)

class TestCasePickleCorrupted(RuntimeError):
    def __bool__(self):
        return False
    __nonzero__ = __bool__

class TestAccelerations(unittest.TestCase):

    @classmethod
    def get_testcase_pickle_name(cls, function_name, testcase_id=1):
        return f"testcase_test_{function_name:s}_{testcase_id:02d}"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        if (hasattr(cls, "test_multi_dimensional_distance")):
            _testcase_01_path = cls.get_testcase_pickle_name("multi_dimensional_distance", testcase_id=1)
            _testcase_01_file = file(_testcase_01_path+".pickle", is_dir=False, script_dir=True)

            if (not _testcase_01_file.isreadable()):
                logging.info (f"Test case 01 for multi_dimensional_distance does not exist yet, generating at {_testcase_01_file.abspath()}")

                _dimensions = 4

                _shape1 = (1000, _dimensions)
                _shape2 = (2000, _dimensions)

                _dtype = np.double

                _input_array1 = np.random.rand(*_shape1) * _shape1[0]
                _input_array2 = np.random.rand(*_shape2) * _shape2[0]

                _output_array = np.empty((_shape1[0], _shape2[0]), dtype = _dtype)

                _device_type = accelerator_type.CPU_MULTIPROCESS

                _process = multi_dimensional_distance.process(type=_device_type)

                with execute_timer(echo=True, report="Test case 01 for multi_dimensional_distance generated in {:,.6f} seconds."):
                    _output_array = _process(
                        input1=_input_array1,
                        input2=_input_array2,
                        dtype=_dtype,
                        memory_limit=2*(2**30),
                        )

                _testcase_01_data = {
                    "input_array1":_input_array1,
                    "input_array2":_input_array2,
                    "output_array":_output_array,
                }

                for _element in _testcase_01_data:
                    if (isinstance(_testcase_01_data[_element], np.ndarray)):
                        np.savetxt(os.path.join(_testcase_01_file.parent().abspath(), f"{_testcase_01_path}.{_element}.csv"), _testcase_01_data[_element], delimiter=",")

                logging.debug (_testcase_01_file.write(pickle.dumps(_testcase_01_data)),)
                logging.info (f"Test case 01 for multi_dimensional_distance saved.")

                del _input_array1
                del _input_array2
                del _output_array
            else:
                logging.debug (f"Test case 01 for multi_dimensional_distance exists at {_testcase_01_file.abspath()}.")

        if (hasattr(cls, "test_bytes_arrays_xor") or \
            hasattr(cls, "test_bytes_operations_xor")):
            _testcase_01_path = cls.get_testcase_pickle_name("bytes_operations_xor", testcase_id=1)
            _testcase_01_file = file(_testcase_01_path+".pickle", is_dir=False, script_dir=True)

            if (not _testcase_01_file.isreadable()):
                _tests = [
                    {
                        "args":{
                            "bytes1":bytes_to_np(b"ABCDE"),
                            "bytes2":bytes_to_np(b"ABCDE"),
                        },
                        "answer":bytes_to_np(b"\x00\x00\x00\x00\x00"),
                    },
                    {
                        "args":{
                            "bytes1":bytes_to_np(b"ABCDEABCDEFGHIJKLM"),
                            "bytes2":bytes_to_np(b"ABCDE"),
                        },
                        "answer":bytes_to_np(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07\x05\x0B\x0D\x0F\x0A\x0E\x0E"),
                    },
                    {
                        "args":{
                            "bytes1":np.arange(256, dtype=np.uint8),
                            "bytes2":np.arange(2, dtype=np.uint8),
                        },
                        "answer":np.repeat(np.arange(128), 2)*2,
                    },
                ]

                # ====================================
                # XOR encrypt and decrypt
                _input_array1 = np.random.randint(0, 255, 32*2**20, dtype=np.uint8 ) # data
                _input_array2 = np.random.randint(0, 255, 256, dtype=np.uint8 )       # hash
                _output_array = bytes_arrays_xor(_input_array1, _input_array2)        # encrypted

                _tests.append(
                    {
                        "args":{
                            "bytes1":_output_array,
                            "bytes2":_input_array2,
                        },
                        "answer":_input_array1,
                    }
                )

                logging.debug (_testcase_01_file.write(pickle.dumps(_tests)),)
                logging.info (f"Test case 01 for bytes_operations_xor saved.")

                del _input_array1
                del _input_array2
                del _output_array
            else:
                logging.debug (f"Test case 01 for bytes_operations_xor exists at {_testcase_01_file.abspath()}.")

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def conduct_tests(
        self,
        func,
        tests:dict,
        ):

        for _test in tests:
            with self.subTest(input=_test["args"]):
                if (issubclass(_test["answer"], Exception) if (isinstance(_test["answer"], type)) else False):
                    with self.assertRaises(Exception) as context:
                        func(
                            **_test["args"]
                        )

                    self.assertTrue(isinstance(context.exception, _test["answer"]))
                elif (isinstance(_test["answer"], type)):
                    self.assertTrue(isinstance(func(**_test["args"]), _test["answer"]))
                elif (isinstance(_test["answer"], np.ndarray)):
                    if (_test["answer"].dtype in (
                        np.float_,
                        np.float16,
                        np.float32,
                        np.float64,
                        np.float128,
                        np.longfloat,
                        np.half,
                        np.single,
                        np.double,
                        np.longdouble,
                    )):
                        _assertion = np.testing.assert_allclose
                    else:
                        _assertion = np.testing.assert_array_equal

                    _assertion(
                        func(
                            **_test["args"]
                        ),
                        _test["answer"],
                    )

                else:
                    self.assertEqual(
                        func(
                            **_test["args"]
                        ),
                        _test["answer"],
                    )

    def test_distance_between_two_points(self) -> None:
        _tests = [
            {
                # Standard Logic test
                "args":{
                    "coor1":np.array([0,0,0,0]),
                    "coor2":np.array([1,1,1,1]),
                    "dimensions":4,
                },
                "answer":2,
            },
            {
                # High Dimensions
                "args":{
                    "coor1":np.full((1000), 500),
                    "coor2":np.full((1000), -500),
                    "dimensions":1000,
                },
                "answer":31622.776601683792,
            },
            {
                # Unequal Dimensions, but within dimensions
                "args":{
                    "coor1":np.full((10), 0),
                    "coor2":np.full((20), 1),
                    "dimensions":10,
                },
                "answer":3.1622776601683795,
            },
            {
                # Unequal Dimensions, but outside of dimensions - ValueError operands could not be broadcast together with shapes (20,) (10,)
                "args":{
                    "coor1":np.full((10), 0),
                    "coor2":np.full((20), 1),
                    "dimensions":20,
                },
                "answer":ValueError,
            },
            {
                # Non-numeric objects passed - np.core._exceptions._UFuncNoLoopError
                "args":{
                    "coor1":"a string",
                    "coor2":np.array([1,1,1,1]),
                    "dimensions":4,
                },
                "answer":np.core._exceptions._UFuncNoLoopError,
            },
            {
                # Non-numpy arrays passed - TypeError
                "args":{
                    "coor1":[0,0,0,0],
                    "coor2":[1,1,1,1],
                    "dimensions":4,
                },
                "answer":TypeError,
            },
        ]

        self.conduct_tests(
            distance_between_two_points,
            _tests,
        )

    def test_multi_dimensional_distance(self) -> None:
        
        cls=type(self)
        _testcase_01_path = cls.get_testcase_pickle_name("multi_dimensional_distance", testcase_id=1)
        _testcase_01_file = file(_testcase_01_path+".pickle", is_dir=False, script_dir=True)

        try:
            _testcase_01_json = pickle.loads(_testcase_01_file.read(output=bytes))

            _input_array1 = _testcase_01_json["input_array1"]
            _input_array2 = _testcase_01_json["input_array2"]
            _output_array = _testcase_01_json["output_array"]

        except (pickle.UnpicklingError, KeyError) as e:
            logging.error (f"Test case 01 found corrupted, this will be deleted. Rerun this test to create a new one.")
            _testcase_01_file.delete()

            raise TestCasePickleCorrupted(f"Test case 01 {_testcase_01_file.abspath()} found corrupted.")

        _array_size1 = _input_array1.nbytes
        _array_size2 = _input_array2.nbytes
        _output_size = _output_array.nbytes

        _dtype = np.double

        logging.info (f"Array 1 of shape {_input_array1.shape} uses {_array_size1:,d} bytes of memory.")
        logging.info (f"Array 2 of shape {_input_array2.shape} uses {_array_size2:,d} bytes of memory.")
        logging.info (f"Output Array of shape {_output_array.shape} uses {_output_size:,d} bytes of memory.")
        logging.info (f"Total memory consumption: {sum((_array_size1, _array_size2, _output_size)):,d} bytes.")

        _tol = 1e-07

        for _device_type in accelerator_type:
            _process = multi_dimensional_distance.process(type=_device_type)
            
            logging.debug (f"Calculating using {_device_type.name}...\n")

            logging.debug (f"Loaded process with parameters:")
            _parameters = inspect.signature(_process).parameters
            for _parameter in _parameters:
                logging.debug (f"    {_parameter:30s}: {_parameters[_parameter].annotation}")

            with execute_timer(echo=True):
                _test_output = _process(
                    input1=_input_array1,
                    input2=_input_array2,
                    dtype=_dtype,
                    memory_limit=16*(2**10),
                    )

            logging.debug (f"Asserting equalness up to tolerance of {_tol:e}...")
            np.testing.assert_allclose(_test_output, _output_array, rtol=_tol)
            logging.debug ("***\n")

    def test_pad_array_with_random(self) -> None:
        _tests =[
            {
                # To test that the seed is still producing the same thing
                "args":{
                    "array":bytes_to_np(b"ABCDEFG"),
                    "length":300,
                },
                "answer":np.array([
                        96, 153,  24, 133, 187, 219, 228, 159,  66, 147,  80, 247, 155,
                        210,  13,  39, 166, 127, 233,  23, 119, 210,  82,  11, 111, 129,
                        126,  77, 164, 238, 211,  50, 129,  40,  92,  69, 157, 177,  47,
                        1, 102, 201,  51,   4,  63, 137,  26,  14, 114, 175,  20,  52,
                        159, 249, 151, 104,  71,  17, 178, 174, 213, 232, 233,  80, 134,
                        103,  54, 135,  85,  79, 153, 162, 116, 225, 123, 152, 159, 125,
                        160,  62,  72, 137, 249,  33,  74,  52,  24, 216, 116, 234, 213,
                        12, 176, 221,  56, 222,  84, 140,  92, 181, 100, 133,   2, 254,
                        233,  22,  31, 241, 154, 158, 196, 210, 114,  42, 126,  35,  61,
                        45, 120, 178, 188, 137,   5, 122, 187,  13, 120,  75,  65,  98,
                        28, 160, 125, 127,  37, 170,  84,  19, 250,  98, 124, 153, 241,
                        227, 107, 172,   9, 170, 209,  40, 143, 181, 219, 204,  46, 103,
                        162,  24, 144,  49, 163,  63,  80, 149,   2, 237,  23, 185, 248,
                        212, 230, 184, 183, 251, 104,  92, 139,  97,   9,  75, 229, 118,
                        45,  94,  99, 245,  78, 160,  49, 100, 201,  16, 232,   3, 120,
                        141, 100, 221, 126, 116,  35,  87, 176,  72, 121, 156, 107,  46,
                        171, 134, 234, 100, 206, 132, 190,  81,  71, 151, 150,   2, 205,
                        81, 104,  16, 111, 243, 156, 138, 182, 252, 101,  61, 131,  88,
                        77, 108, 231, 134, 205,   9,  36,  80, 134,  12,  62, 216,  14,
                        164, 177, 222, 110,  94, 233,  14, 210, 218,  30, 150,  48,  82,
                        240, 244,   9,  27, 168,  10,  26, 199, 169,  38,  26, 205,   0,
                        168, 151, 205, 169,  22, 229,  43, 120, 249, 167, 253,  61,  61,
                        179, 232,  29, 206, 222,  84, 166, 124, 143, 206, 116, 110, 156,
                        239])
            },
            {
                "args":{
                    "array":bytes_to_np(b"\x00"),
                    "length":24,
                },
                "answer":np.array([
                    94, 129, 193, 216, 206, 234,  14, 162,  32, 214, 216, 129, 247,
                    188,  15,  68, 183, 231, 204,  77, 168,  60, 124,   9])
            },
            {
                # Test of random elements
                "args":{
                    "array":bytes_to_np({ "sample_dict":True, "some_data":[None, False, 234.567] }),
                    "length":32,
                },
                "answer":np.array([
                    126, 178, 134,  17, 245, 109,  84, 146, 237, 198,  61,  54, 244,
                    66, 179, 171, 101,  20, 252,  68, 127, 134, 151, 183, 224, 124,
                    241,  99, 219,  43,  25, 143])
            },
        ]

        self.conduct_tests(
            pad_array_with_random,
            _tests,
        )

    def setUp_bytes_xor(self) -> None:
        
        cls=type(self)
        _testcase_01_path = cls.get_testcase_pickle_name("bytes_operations_xor", testcase_id=1)
        _testcase_01_file = file(_testcase_01_path+".pickle", is_dir=False, script_dir=True)

        try:
            _testcase_01_json = pickle.loads(_testcase_01_file.read(output=bytes))

            self.testcase_bytes_xor = _testcase_01_json

        except (pickle.UnpicklingError, KeyError) as e:
            logging.error (f"Test case 01 found corrupted, this will be deleted. Rerun this test to create a new one.")
            _testcase_01_file.delete()

            raise TestCasePickleCorrupted(f"Test case 01 {_testcase_01_file.abspath()} found corrupted.")

    def _test_bytes_arrays_xor(self) -> None:
        self.setUp_bytes_xor()
        _tests = self.testcase_bytes_xor

        # ====================================

        self.conduct_tests(
            bytes_arrays_xor,
            _tests,
        )

    def test_bytes_operations_xor(self) -> None:
        self.setUp_bytes_xor()
        _tests = self.testcase_bytes_xor

        for _device_type in accelerator_type:
            _process = bytes_XOR.process(type=_device_type)

            logging.debug (f"Calculating using {_device_type.name}...\n")

            logging.debug (f"Loaded process with parameters:")
            _parameters = inspect.signature(_process).parameters
            for _parameter in _parameters:
                logging.debug (f"    {_parameter:30s}: {_parameters[_parameter].annotation}")
            
            with execute_timer(echo=True):
                self.conduct_tests(
                    _process,
                    _tests
                )

            logging.debug ("***\n")

    def test_bytes_operations_rol(self) -> None:
        pass

    def test_bytes_operations_ror(self) -> None:
        pass

    def test_bytes_operations_shl(self) -> None:
        pass

    def test_bytes_operations_shr(self) -> None:
        pass

    def test_geodistance_sphr_between_two_latlngs(self) -> None:
        _tests = [
            {
                "args": {
                    "s_lat": 0,
                    "s_lng": 0,
                    "e_lat": 0,
                    "e_lng": 90
                },
                "answer": 10010.684990663874
            },
            {
                "args": {
                    "s_lat": 51.50737491590355,
                    "s_lng": -0.12703183497203677,
                    "e_lat": 50.95131148321214,
                    "e_lng": 1.8593262909456834
                },
                "answer": 151.54418168316775
            },
            {
                "args": {
                    "s_lat": 22.29421093707705,
                    "s_lng": 114.16912196400331,
                    "e_lat": 0,
                    "e_lng": 0
                },
                "answer": 12486.767591102758
            },
            {
                "args": {
                    "s_lat": 22.29421093707705,
                    "s_lng": 114.16912196400331,
                    "e_lat": 40.68925065660414,
                    "e_lng": -74.04450653589767
                },
                "answer": 12964.353453784852
            },
            {
                "args": {
                    "s_lat": 22.29421093707705,
                    "s_lng": -245.8308780359967,
                    "e_lat": 40.68925065660414,
                    "e_lng": -74.04450653589767
                },
                "answer": 12964.353453784852
            },
            {
                "args": {
                    "s_lat": 0,
                    "s_lng": -179.9999,
                    "e_lat": 0,
                    "e_lng": 179.9999
                },
                "answer": 0.02224596664487551
            },
            {
                "args": {
                    "s_lat": 89,
                    "s_lng": 0,
                    "e_lat": -89,
                    "e_lng": 0
                },
                "answer": 19798.910314868575
            }
        ]

        self.conduct_tests(
            njit_geodistance_sphr_between_two_latlngs,
            _tests,
        )

    def test_geodistance_ellip_between_two_latlngs(self) -> None:
        _tests = [
            {
                "args": {
                    "s_lat": 0,
                    "s_lng": 0,
                    "e_lat": 0,
                    "e_lng": 90
                },
                "answer": 10018.754171390094
            },
            {
                "args": {
                    "s_lat": 51.50737491590355,
                    "s_lng": -0.12703183497203677,
                    "e_lat": 50.95131148321214,
                    "e_lng": 1.8593262909456834
                },
                "answer": 151.90920889884373
            },
            {
                "args": {
                    "s_lat": 22.29421093707705,
                    "s_lng": 114.16912196400331,
                    "e_lat": 0,
                    "e_lng": 0
                },
                "answer": 12495.209263526498
            },
            {
                "args": {
                    "s_lat": 22.29421093707705,
                    "s_lng": 114.16912196400331,
                    "e_lat": 40.68925065660414,
                    "e_lng": -74.04450653589767
                },
                "answer": 12980.172222211004
            },
            {
                "args": {
                    "s_lat": 22.29421093707705,
                    "s_lng": -245.8308780359967,
                    "e_lat": 40.68925065660414,
                    "e_lng": -74.04450653589767
                },
                "answer": 12980.172222211004
            },
            {
                "args": {
                    "s_lat": 0,
                    "s_lng": -179.9999,
                    "e_lat": 0,
                    "e_lng": 179.9999
                },
                "answer": 0.022263897320092842
            },
            {
                "args": {
                    "s_lat": 89,
                    "s_lng": 0,
                    "e_lat": -89,
                    "e_lng": 0
                },
                "answer": 19780.54372879477
            }
        ]

        self.conduct_tests(
            njit_geodistance_ellip_between_two_latlngs,
            _tests,
        )

    def test_geodistance(self)->None:
        # Load data from files
        _testcase_01_filenames = {
            "input_array1":"testcase_test_geodistance_01.input_array1.txt",
            "input_array2":"testcase_test_geodistance_01.input_array2.txt",
            "output_haversine":"testcase_test_geodistance_01_haversine.output_array.txt",
            "output_vincenty":"testcase_test_geodistance_01_vincenty.output_array.txt",
        }

        def _loadtxt(path, cols):
            with open(path, "r") as _f:
                return np.loadtxt(_f, delimiter=",").reshape((-1, cols))
        
        _testcase_01_data = {
            _key:_loadtxt(_testcase_01_filenames[_key],2) for _key in ("input_array1", "input_array2")
        }

        for _key in ("output_haversine", "output_vincenty"):
            _testcase_01_data[_key] = _loadtxt(_testcase_01_filenames[_key], _testcase_01_data["input_array2"].shape[0])

        # Put together a list of tests
        _tests = [
            {
                "args":{
                    "input1":_testcase_01_data["input_array1"],
                    "input2":_testcase_01_data["input_array2"],
                    "max_dist":-1,
                    "precise":False,
                },
                "answer":_testcase_01_data["output_haversine"],
            },
            {
                "args":{
                    "input1":_testcase_01_data["input_array1"],
                    "input2":_testcase_01_data["input_array2"],
                    "max_dist":-1,
                    "precise":True,
                },
                "answer":_testcase_01_data["output_vincenty"],
            },
        ]

        with self.subTest(method="process_cpu_parallel"):
            self.conduct_tests(
                geodistance.process_cpu_parallel,
                _tests,
            )

        if (numba and cuda):
            with self.subTest(method="process_cuda"):
                self.conduct_tests(
                    geodistance.process_cuda,
                    _tests,
                )
        

if __name__ == "__main__":
    unittest.main()