import onnx
# import onnxruntime
from onnx import numpy_helper
import subprocess
import numpy as np
import os
from datetime import datetime

from onnx import helper, TensorProto, ModelProto, OptionalProto
from onnx import onnx_pb as onnx_proto
import itertools
import struct
from typing import Optional

model_path = "./faster_rcnn_R_50_FPN_1x.onnx"
model_path = "./augmented.onnx"
model_path = "./TensorrtExecutionProvider_TRTKernel_graph_torch-jit-export_11340214133003188726_252_46.onnx"
model_path = "./wbs_model/Model.onnx"
model_path = "/home/azureuser/tiny-yolov3/yolov3-tiny.onnx"
# model_path = "/home/azureuser/perf/scan_model/BLSTM_i64_h128_x2_scan.onnx"
model_path = "/home/azureuser/perf/v3_smaller/scale_x2_model_v3_small_rand-shape.onnx"
model_path = "/home/azureuser/perf/CTFM/r9b_ctfm.untrained.20220308.fp32.prep.infer.fp16.infer.onnx"
model_path = "/home/azureuser/mnt/roberta/roberta_qdq_model.onnx"
# model_path = "/home/azureuser/mnt/roberta/model_quantized.onnx"
# model_path = "/home/azureuser/mnt/roberta/model_quantized_2.onnx"
test_input_path = "./"
test_input_path = "./wbs_model"
test_input_path = "/home/azureuser/tiny-yolov3/test_data_set_0"
test_input_path = "/home/azureuser/perf/v3_smaller/test_data_set_0"
# test_input_path = "/home/azureuser/perf/scan_model/test_data_set_0"
test_input_path = "/home/azureuser/mnt/roberta/test_data_set_0"
providers = [    
    ("TensorrtExecutionProvider", {
        "trt_int8_enable": True,
    }),
    ("CUDAExecutionProvider")
]
# providers = [{"trt_int8_enable": True}]
# providers = ["CPUExecutionProvider"]
is_random_input = False 
num_random_input = 100 # use only when is_random_input is True

os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "12097483648" 

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

def generate_and_save_tensor_to_file_2():
    shape = (1, 128)
    file_dir = "test_data_set_0/" 

    dtype = np.int64
    np_array = np.random.randint(0, 1000, shape, dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_0.bin"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    dtype = np.int64
    np_array = np.random.randint(0, 1000, shape, dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_1.bin"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    dtype = np.int64
    np_array = np.random.randint(0, 1000, shape, dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_2.bin"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

def read_from_npy_and_convert_to_proto():
    np_array = np.load("2867_Reshaped.npy")
    tensor = numpy_helper.from_array(np_array)
    file_name = "2867_Reshaped.pb" 
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

def get_float32_random_np_array(shape, save_to_pb, index):
    dtype = np.float32
    np_array = np.random.random_sample(shape).astype(dtype)

    if save_to_pb:
        tensor = numpy_helper.from_array(np_array)
        pb_name = "input_" + str(index) +".pb"
        with open(pb_name, 'wb') as f:
            f.write(tensor.SerializeToString())

    return np_array

def get_inputs_for_ctfm_model(seq, batch):
    inputs = {}
    save_to_pb = True
    index = 0

    # 'input'
    name = "input"
    shape = (seq, batch, 160)
    inputs[name] = get_float32_random_np_array(shape, save_to_pb, index)
    index += 1

    # 'SequenceStartFlag' 
    name = "SequenceStartFlag"
    shape = (batch, 1)
    np_array = np.random.choice([True, False], size=shape) 
    inputs[name] = np_array 
    if save_to_pb:
        tensor = numpy_helper.from_array(np_array)
        pb_name = "input_" + str(index) +".pb"
        with open(pb_name, 'wb') as f:
            f.write(tensor.SerializeToString())
    index += 1

    # 'ctx.0'
    name = "ctx.0"
    shape = (batch, 8, 160)
    inputs[name] = get_float32_random_np_array(shape, save_to_pb, index)
    index += 1

    # 'ctx.1' to 'ctx.18'
    for i in range(1, 19):
        name = "ctx." + str(i)
        shape = (batch, 346, 624) 
        inputs[name] = get_float32_random_np_array(shape, save_to_pb, index)
        index += 1

    # 'in_history_length'
    name = "in_history_length"
    shape = (batch, 1)
    inputs[name] = get_float32_random_np_array(shape, save_to_pb, index)
    index += 1

    return inputs

def generate_and_save_tensor_to_file_fluency_model_subgraph():

    file_dir = "test_data_set_fluency_subgraph/" 
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    ## input 1
    dtype = np.bool_
    np_array = np.asarray(False)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_0.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "use_past.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)


    ## input 2
    dtype = np.int64
    np_array = np.asarray([1, 32, 512])
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_1.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "1972.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)


    ## input 3 & 4
    dtype = np.float32
    shape = (1, 8, 1, 64) 
    np_array = np.random.random_sample(shape).astype(dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_2.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "1911.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)

    file_name = file_dir + "input_3.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "1909.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)

    ## input 5 
    dtype = np.float32
    shape = (1, 32, 512) 
    np_array = np.random.random_sample(shape).astype(dtype)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_4.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "query.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)

    ## input 6 
    dtype = np.int64
    np_array = np.asarray(32)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_5.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "1971.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)


    ## input 7 
    dtype = np.int64
    np_array = np.asarray([32, 8, -1, 64])
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_6.pb"

    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open(file_dir + "2070.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)

def generate_and_save_tensor_to_file():
    # generate input for one of faster-rcnn subgraph
    shape_1 = (1000, 324)
    shape_1 = (3, 3, 32, 32)
    shape_1 = (1, 4, 64)
    shape_1 = (1, 2048)
    shape_1 = (16, 1)
    shape_1 = (1, 1)


    # shape_2 = (0, 3)
    shape_2 = (1, 59)
    shape_2 = (4)
    shape_2 = (1, 256)
    # shape_2 = (4, 77)
    # shape_3 = (0, 4)
    # shape_3 = (1, 59)
    shape_2 = (4, 77)
    shape_2 = (1, 256, 8, 8)

    shape_3 = (1, 3, 1600, 1600)
    shape_3 = (1, 3, 256, 256)
    shape_3 = (1, 1, 1)
    shape_3 = (1, 4, 2)
    shape_3 = (1,)
    shape_3 = (16, 1, 160)
    shape_3 = (1, 3, 2)
    # shape_3 = (1, 346, 624)
    # shape_3 = (1, 2)
    # shape_3 = (2, 3, 800, 800)
    # shape_3 = (1, 3, 90, 90)
    # shape_3 = (500, 64, 64)
    # shape_3 = (1, 8, 1, 64)
    # shape_3 = (16, 3, 224, 224)
    shape_3 = (12, 256, 256)
    shape_3 = (1, 4, 512, 512)
    shape_3 = (1, 12, 256, 64)
    shape_3 = (1, 256, 768)
    shape_3 = (2, 2)

    # dtype = np.float32
    # dtype = np.int64
    dtype = np.bool_
    shape = shape_1
    # np_array = np.random.random_sample(shape).astype(dtype)
    # np_array = np.random.randint(0, 1000, shape, dtype)
    np_array = np.random.choice(a=[False, True], size=shape)
    # np_array = np.asarray(False)
    tensor = numpy_helper.from_array(np_array)
    file_dir = "test_data_set_0/" 
    file_name = file_dir + "input_0.pb"

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(file_name, 'wb') as f:
        f.write(tensor.SerializeToString())

    with open("2837.bin", "wb") as f:
        if np_array.size != 0:
            for x in np.nditer(np_array):
                a=struct.pack('f',x)
                f.write(a)


    dtype = np.int32
    # dtype = np.int64
    shape = shape_2
    # np_array = np.random.randint(0, 1000, shape, dtype)
    # np_array = np.asarray([32, 8, -1, 64])
    np_array = np.asarray(8)
    tensor = numpy_helper.from_array(np_array)
    file_name = file_dir + "input_1.pb"
    print(np_array)

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
    # np_array = np.random.randint(0, 1000, shape, dtype)
    tensor = numpy_helper.from_array(np_array)

    # # values = [1.1, 2.2, 3.3, 4.4, 5.5]
    # # values_tensor = helper.make_tensor(
        # # name="test", data_type=TensorProto.FLOAT, dims=(5,), vals=values
    # # )

    # optional = helper.make_optional(
       # name="test", elem_type=OptionalProto.TENSOR, value=tensor
    # )
    # tensor = optional

    file_name = file_dir + "input_2.pb"

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

def run_ctfm_model():
    onnxruntime.set_default_logger_severity(0)
    options = onnxruntime.SessionOptions()
    # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    # options.log_severity_level = 0 
    session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=options)
    session.enable_fallback()

    inputs = get_inputs_for_ctfm_model(64, 1)
    print(inputs)
    result = session.run(None, inputs)

    print("============================= inference done =============================")
    inputs = get_inputs_for_ctfm_model(64, 8)
    # print(inputs)
    result = session.run(None, inputs)

# # augment_model(model_path)
# read_from_npy_and_convert_to_proto()
# exit(0)
# generate_and_save_tensor_to_file_fluency_model_subgraph()

generate_and_save_tensor_to_file()
exit(0)

# inputs = get_inputs_for_ctfm_model(128, 1)
# print(inputs)
# run_ctfm_model()
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
            # with open(bin, 'rb') as f:
                # tensor.ParseFromString(f.read())
                # tensor_to_array = numpy_helper.to_array(tensor)
                # tensor_to_array = np.reshape(tensor_to_array, (1, 3, 384, 288))
                # tensor_to_array = tensor_to_array.astype(np.single)
                # input_data.append(tensor_to_array)
            if os.stat(bin).st_size == 0:
                print(bin)
                if "input_1" in bin:
                    input_data.append(np.asarray([], dtype=np.int64).reshape(0, 3))
                elif "input_2" in bin:
                    input_data.append(np.asarray([], dtype=np.float32).reshape(0, 4))
            else:
                tensor = onnx.TensorProto()
                with open(bin, 'rb') as f:
                    tensor.ParseFromString(f.read())
                    tensor_to_array = numpy_helper.to_array(tensor)
                    input_data.append(tensor_to_array)

        print(input_data)
        input_data_list.append(input_data)

print(model_path)

# session creation
# options.log_severity_level = 0 
onnxruntime.set_default_logger_severity(0)
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=options)
print(session.get_providers())
# print(os.listdir())


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
