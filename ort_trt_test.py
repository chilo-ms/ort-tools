import onnx
import onnxruntime
from onnx import numpy_helper
import subprocess
import numpy as np
import os
from datetime import datetime

from onnx import helper, TensorProto, ModelProto
from onnx import onnx_pb as onnx_proto
import itertools
import struct

model_path = "./faster_rcnn_R_50_FPN_1x.onnx"
model_path = "./augmented.onnx"
model_path = "./TensorrtExecutionProvider_TRTKernel_graph_torch-jit-export_11340214133003188726_252_46.onnx"
test_input_path = "./"
providers = ["TensorrtExecutionProvider"]
# providers = ["CUDAExecutionProvider"]
# providers = ["CPUExecutionProvider"]
is_random_input = False 
num_random_input = 100 # use only when is_random_input is True

os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"

def split_and_sort_output(string_list):
    string_list = string_list.split("\n")
    string_list.sort()
    return string_list

def run_command(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output = p.stdout.decode("ascii").strip()
    return output

# this function only for faster-rcnn
def generate_random_input(shape):
    dtype = np.float32
    tensor = np.random.random_sample(shape).astype(dtype)
    return tensor

def generate_and_save_tensor_to_file():
    # generate input for one of faster-rcnn subgraph
    shape_1 = (1000, 324)
    shape_2 = (0, 3)
    shape_3 = (0, 4)

    dtype = np.float32
    shape = shape_1
    np_array = np.random.random_sample(shape).astype(dtype)
    tensor = numpy_helper.from_array(np_array)
    file_dir = "test_data_set_0/" 
    file_name = file_dir + "input_0.bin"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open("2837.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)


    dtype = np.int64
    shape = shape_2
    np_array = np.random.randint(0, 1000, shape, dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_1.bin"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open("2873.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)


    dtype = np.float32
    shape = shape_3
    np_array = np.random.random_sample(shape).astype(dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_2.bin"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open("2867.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)

def select_tensors_to_calibrate(model, op_types_to_calibrate, tensor_name_to_calibrate):
    '''
    select all quantization_candidates op type nodes' input/output tensors. 
    returns:
        tensors (set): set of tensor name.
        value_infos (dict): tensor name to value info.
    '''
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({ot.name: ot for ot in model.graph.output})
    value_infos.update({it.name: it for it in model.graph.input})
    initializer = set(init.name for init in model.graph.initializer)

    tensors_to_calibrate = set()
    tensor_type_to_calibrate = set([TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.INT32, TensorProto.INT64])

    for node in model.graph.node:
        if len(tensor_name_to_calibrate) > 0:
            for tensor_name in itertools.chain(node.input, node.output):
                if tensor_name in value_infos.keys() and tensor_name in tensor_name_to_calibrate:
                    vi = value_infos[tensor_name]
                    print(tensor_name)
                    print(vi)
                    if vi.type.HasField('tensor_type') and (
                            vi.type.tensor_type.elem_type in tensor_type_to_calibrate) and (
                                tensor_name not in initializer):
                        tensors_to_calibrate.add(tensor_name)
        else:
            if len(op_types_to_calibrate) == 0 or node.op_type in op_types_to_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos.keys():
                        vi = value_infos[tensor_name]
                        if vi.type.HasField('tensor_type') and (
                                vi.type.tensor_type.elem_type in tensor_type_to_calibrate) and (
                                    tensor_name not in initializer):
                            tensors_to_calibrate.add(tensor_name)

    print(tensors_to_calibrate)
    # print(value_infos.keys())
    return tensors_to_calibrate, value_infos

def augment_model(model_path):
        added_nodes = []
        added_outputs = []
        model = onnx.load(model_path)
        tensor_name_to_calibrate = ["2867", "2876", "2873", "2837"]
        op_types_to_calibrate = []
        tensors, value_infos = select_tensors_to_calibrate(model, op_types_to_calibrate, tensor_name_to_calibrate) 

        for tensor in tensors:
            added_outputs.append(value_infos[tensor])

        model.graph.node.extend(added_nodes)
        model.graph.output.extend(added_outputs)
        onnx.save(model, "augmented.onnx", save_as_external_data=False)
        # self.augment_model = model

# augment_model(model_path)
# generate_and_save_tensor_to_file()
# exit(0)

input_data_list = []
if is_random_input:
    # generate random input
    for count in range(num_random_input): 
        shape = (3, 800, 1088)
        input_data_list.append([generate_random_input(shape)])
else:
    # parse existed input protobuf
    output = run_command(["find", test_input_path, "-name", "test_data_set_*"])
    result = split_and_sort_output(output)
    print(result)
    for path in result:
        bin_output = run_command(["find", path, "-name", "input*"])
        bin_result = split_and_sort_output(bin_output)
        print(bin_result)
        input_data = []
        for bin in bin_result:
            tensor = onnx.TensorProto()
            with open(bin, 'rb') as f:
                tensor.ParseFromString(f.read())
                tensor_to_array = numpy_helper.to_array(tensor)
                input_data.append(tensor_to_array)
            # if os.stat(bin).st_size == 0:
                # print(bin)
                # if "input_1" in bin:
                    # input_data.append(np.asarray([], dtype=np.int64).reshape(0, 3))
                # elif "input_2" in bin:
                    # input_data.append(np.asarray([], dtype=np.float32).reshape(0, 4))
            # else:
                # tensor = onnx.TensorProto()
                # with open(bin, 'rb') as f:
                    # tensor.ParseFromString(f.read())
                    # tensor_to_array = numpy_helper.to_array(tensor)
                    # input_data.append(tensor_to_array)

        print(input_data)
        input_data_list.append(input_data)

# session creation
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=options)
print(session.get_providers())


# run inference session
latency = []
for num in range(len(input_data_list)):
    print("-------------------------------------------------")
    print(num)
    session_inputs = {}
    session_outputs = []
    for i in range(len(session.get_inputs())):
        session_inputs[session.get_inputs()[i].name] = input_data_list[num][i]
    for i in range(len(session.get_outputs())):
        session_outputs.append(session.get_outputs()[i].name)

    perf_start_time = datetime.now()
    result = session.run(session_outputs, session_inputs)
    perf_end_time = datetime.now()
    latency.append(perf_end_time - perf_start_time)
    print(result)
    # result1 = result[0]
    # with open("2837.bin", "wb") as f:
        # if result1.size != 0:
            # for x in np.nditer(result1):
                # a=struct.pack('f',x)
                # f.write(a)

    # result2 = result[1]
    # with open("2873.bin", "wb") as f:
        # if result2.size != 0:
            # for x in np.nditer(result2):
                # a=struct.pack('f',x)
                # f.write(a)

    # result3 = result[2]
    # with open("2867.bin", "wb") as f:
        # if result3.size != 0:
            # for x in np.nditer(result3):
                # a=struct.pack('f',x)
                # f.write(a)

    # print(latency)
    # # print("\nTotal time for inference: {}".format(perf_end_time - perf_start_time))
