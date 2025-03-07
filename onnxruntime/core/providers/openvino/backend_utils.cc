// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#include "ov_interface.h"
#include <ngraph/pass/convert_fp32_to_fp16.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include "core/providers/shared_library/provider_api.h"
#include "backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {
namespace backend_utils {

#ifndef NDEBUG
bool IsDebugEnabled() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_DEBUG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}
void DumpOnnxModelProto(const ONNX_NAMESPACE::ModelProto& model_proto, std::string file_name) {
  std::fstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
  model_proto.SerializeToOstream(outfile);
}

#endif

bool IsCILogEnabled() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

bool UseCompiledNetwork() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("OV_USE_COMPILED_NETWORK");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

std::string GetCurrentWorkingDir() {
  std::string curr_dir;
  ORT_UNUSED_PARAMETER(curr_dir);
  char buff[FILENAME_MAX];
  curr_dir = GetCurrentDir(buff, FILENAME_MAX);
  std::string current_working_dir(buff);
  return current_working_dir;
}

bool IsDirExists(const std::string& pathname) {
  struct stat info;
  if(stat(pathname.c_str(), &info) != 0) {
    LOGS_DEFAULT(INFO) << log_tag << "cannot access pathname: " << pathname;
	  return false;
  } else if(info.st_mode & S_IFDIR) {
      LOGS_DEFAULT(INFO) << log_tag << "pathname exists: " << pathname;
	    return true;
  } else {
      LOGS_DEFAULT(INFO) << log_tag << "pathname: " << pathname << ": doesn't contain the directory 'ov_compiled_blobs' ";
  }
  return false;
}

void CreateDirectory(const std::string& ov_compiled_blobs_dir) {
  LOGS_DEFAULT(INFO) << log_tag << "'ov_compiled_blobs' directory doesn't exist at the executable path, so creating one";
#if defined(_WIN32)
  if (_mkdir(ov_compiled_blobs_dir.c_str()) == 0) {  // Creating a directory
	  LOGS_DEFAULT(INFO) << log_tag << "created a directory named 'ov_compiled_blobs' at the executable path";
  } else {
    LOGS_DEFAULT(INFO) << log_tag << "Error creating a directory named 'ov_compiled_blobs' at the executable path";
    throw std::runtime_error("Could not create the directory");
  }
#else
  if (mkdir(ov_compiled_blobs_dir.c_str(), 0777) == 0) { // Creating a directory
    LOGS_DEFAULT(INFO) << log_tag << "created a directory named 'ov_compiled_blobs' at the executable path";
  } else {
    LOGS_DEFAULT(INFO) << log_tag << "Error creating a directory named 'ov_compiled_blobs' at the executable path";
    throw std::runtime_error("Could not create the directory");
  }
#endif
}

struct static_cast_int64 {
  template <typename T1>  // T1 models type statically convertible to T
  int64_t operator()(const T1& x) const { return static_cast<int64_t>(x); }
};

std::shared_ptr<InferenceEngine::CNNNetwork>
CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context, const SubGraphContext& subgraph_context, std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map) {
  if(IsCILogEnabled()) {
    std::cout << "CreateNgraphFunc" << std::endl;
  }

#ifndef NDEBUG
  if (IsDebugEnabled()) {
    DumpOnnxModelProto(model_proto, subgraph_context.subgraph_name + "_static.onnx");
  }
#endif

  std::shared_ptr<ngraph::Function> ng_function;
  #if defined (OPENVINO_2021_4)
    const std::string model = model_proto.SerializeAsString();
    auto cnn_network = global_context.ie_core.ReadModel(model);
    ng_function = cnn_network->getFunction();
  #else
     ORT_UNUSED_PARAMETER(model_proto);
  #endif

  if (global_context.device_type.find("GPU") != std::string::npos &&
      subgraph_context.precision == InferenceEngine::Precision::FP16) {
    //FP16 transformations
    ngraph::pass::ConvertFP32ToFP16().run_on_function(ng_function);
    ng_function->validate_nodes_and_infer_types();
  }

  if (!global_context.is_wholly_supported_graph) {
    std::map<std::string, std::string> result_to_output;
    for (auto& result : ng_function->get_results()) {
      result_to_output[result->get_friendly_name()] = result->input_value(0).get_node_shared_ptr()->get_friendly_name();
    }
    ngraph::pass::ConstantFolding().run_on_function(ng_function);
    auto& results = const_cast<::ngraph::ResultVector&>(ng_function->get_results());
    size_t index = results.size() - 1;
    #if defined (OV_API_20)
      for (auto it = results.rbegin(); it != results.rend(); ++it) {
      if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>((*it)->input_value(0).get_node_shared_ptr())) {
        const_outputs_map[(*it)->get_friendly_name()] = const_node;
        results.erase(results.begin() + index);
      }
      --index;
    }
    #else
      for (auto it = results.rbegin(); it != results.rend(); ++it) {
        if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>((*it)->input_value(0).get_node_shared_ptr())) {
          const_outputs_map[result_to_output.at((*it)->get_friendly_name())] = const_node;
          results.erase(results.begin() + index);
        }
        --index;
      }
    #endif
  }

  return std::make_shared<InferenceEngine::CNNNetwork>(ng_function);
};

#if defined (OV_API_20)
std::shared_ptr<OVNetwork>
CreateOVModel(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context, const SubGraphContext& subgraph_context, std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map) {
  if(IsCILogEnabled()) {
    std::cout << "CreateNgraphFunc" << std::endl;
  }

#ifndef NDEBUG
  if (IsDebugEnabled()) {
    DumpOnnxModelProto(model_proto, subgraph_context.subgraph_name + "_static.onnx");
  }
#endif

  const std::string model = model_proto.SerializeAsString();
  auto cnn_network = global_context.ie_core.ReadModel(model);

  if ((subgraph_context.precision == InferenceEngine::Precision::FP16) &&
      (global_context.device_type.find("MYRIAD") == std::string::npos)) {
    //FP16 transformations
    ov::pass::ConvertFP32ToFP16 pass_obj;
    pass_obj.run_on_model(cnn_network);
    cnn_network->validate_nodes_and_infer_types();

    auto proc = ov::preprocess::PrePostProcessor(cnn_network);
    for (size_t i=0; i < cnn_network->inputs().size(); i++) {
      if(cnn_network->inputs()[i].get_element_type() == ov::element::f16) {
        proc.input(i).tensor().set_element_type(ov::element::f32);
        proc.input(i).preprocess().convert_element_type(ov::element::f16);
      }
    }

    for (size_t i=0; i < cnn_network->outputs().size(); i++) {
      if(cnn_network->outputs()[i].get_element_type() == ov::element::f16) {
        proc.output(i).postprocess().convert_element_type(ov::element::f32);
      }
    }
    cnn_network = proc.build();
  }

  //Check for Constant Folding
  if (!global_context.is_wholly_supported_graph) {
    ov::pass::ConstantFolding pass_const_obj;
    pass_const_obj.run_on_model(cnn_network);
    auto& results = const_cast<ov::ResultVector&>(cnn_network.get()->get_results());
    size_t index = results.size() - 1;

    for (auto it = results.rbegin(); it != results.rend(); ++it) {
      if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>((*it)->input_value(0).get_node_shared_ptr())) {
        const_outputs_map[(*it)->get_friendly_name()] = const_node;
        results.erase(results.begin() + index);
      }
      --index;
    }
  }
  return cnn_network;
}
#endif

InferenceEngine::Precision ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type, std::string device) {
  ONNX_NAMESPACE::DataType type_string = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(onnx_type);
  if (*type_string == "float" || *type_string == "tensor(float)") {
    return InferenceEngine::Precision::FP32;
  } else if (*type_string == "float16" || *type_string == "tensor(float16)") {
    return InferenceEngine::Precision::FP16;
  } else if (*type_string == "int32" || *type_string == "tensor(int32)") {
    return InferenceEngine::Precision::I32;
  } else if (*type_string == "int16" || *type_string == "tensor(int16)") {
    return InferenceEngine::Precision::I16;
  } else if (*type_string == "int8" || *type_string == "tensor(int8)") {
    return InferenceEngine::Precision::I8;
  } else if (*type_string == "uint16" || *type_string == "tensor(uint16)") {
    return InferenceEngine::Precision::U16;
  } else if (*type_string == "uint8" || *type_string == "tensor(uint8)") {
    return InferenceEngine::Precision::U8;
  } else if (*type_string == "bool" || *type_string == "tensor(bool)") {
    if (device == "MYRIAD") {
      return InferenceEngine::Precision::I32;
    } else {
      return InferenceEngine::Precision::U8;
    }
  } else if (*type_string == "int64" || *type_string == "tensor(int64)") {
    return InferenceEngine::Precision::I32;
  } else {
    ORT_THROW(log_tag + "Unsupported Data type");
  }
}

void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network,
               std::unordered_map<std::string, int> output_names,
               std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map,
               std::string device) {
  // Configure input & output
  // Prepare input blobs

  auto inputInfo = network->getInputsInfo();
  int input_idx = 0;
  for (auto iter = inputInfo.begin(); iter != inputInfo.end(); ++iter, ++input_idx) {
    // Get the onnx index for the corresponding input (ignoring initializers)
    auto precision = ConvertPrecisionONNXToOpenVINO(model_proto.graph().input(input_idx).type(), device);
    iter->second->setPrecision(precision);
  }

  // Prepare output blobs
  auto outputInfo = network->getOutputsInfo();
  for (auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter) {
    auto output_name = iter->first;
    auto it = const_outputs_map.find(output_name);
    //Output is constant and don't need to set precision
    if (it != const_outputs_map.end())
      break;
    auto itr = output_names.find(output_name);
    if (itr == output_names.end()) {
      ORT_THROW(log_tag + "Output Names Mismatch: " + output_name + " doesn't exist");
    }
    auto precision = ConvertPrecisionONNXToOpenVINO(model_proto.graph().output(itr->second).type(), device);
    iter->second->setPrecision(precision);
  }
}

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context, size_t batch_size,
                OVInferRequestPtr infer_request,
                std::string output_name,
                std::unordered_map<std::string, int> output_names) {
  OrtValue* output_tensor;
  auto graph_output_blob = infer_request->GetTensor(output_name);

  #if defined (OV_API_20)
  auto graph_output_dims = graph_output_blob->get_shape();
  #else
  auto graph_output_dims = graph_output_blob->getTensorDesc().getDims();
  #endif

  if (batch_size > 1) {
    // Add the batch size as dim 0.
    graph_output_dims.insert(graph_output_dims.begin(), batch_size);
  }
  size_t num_dims = graph_output_dims.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(graph_output_dims[j]);
  }
  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  output_tensor = ort.KernelContext_GetOutput(context, index, output_shape.get(), num_dims);
  return output_tensor;
}

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context,
                std::string output_name,
                std::unordered_map<std::string, int> output_names,
                std::shared_ptr<ngraph::Node> node) {
  OrtValue* output_tensor;

  #if (defined OV_API_20)
    // Find position of '/' in the output_name
    int pos = output_name.find("/");
    // Copy the substring from start to pos
    output_name = output_name.substr(0 , pos);
  #endif

  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  auto shape = node->get_shape();

  size_t num_dims = shape.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(shape[j]);
  }
  output_tensor = ort.KernelContext_GetOutput(context, index, output_shape.get(), num_dims);

  return output_tensor;
}

int GetFirstAvailableDevice(GlobalContext& global_context) {
  int i = 0;
  //Get the first available VAD-M device and set the device to busy
  while (i < 8) {
    bool device = global_context.deviceAvailableList[i];
    if (device) {
      global_context.deviceAvailableList[i] = false;
      break;
    }
    i++;
  }
  //If all of the devices are busy, assign the first device and
  //make all remaining devices free
  if (i == 8) {
    i = 0;
    global_context.deviceAvailableList[i] = false;
    for (int j = 1; j < 8; j++) {
      global_context.deviceAvailableList[j] = true;
    }
  }
  return i;
}

void FillOutputsWithConstantData(Ort::CustomOpApi& ort, std::shared_ptr<ngraph::Node> node, OrtValue* out_tensor) {
  switch (node->get_element_type()) {
    case ngraph::element::Type_t::f32: {
      FillOutputHelper<float>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::boolean: {
      FillOutputHelper<char>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::i32: {
      FillOutputHelper<int32_t>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::i64: {
      FillOutputHelper<int64_t>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::f16: {
      FillOutputHelper<float>(ort, out_tensor, node);
      break;
    }
    default:
      ORT_THROW(log_tag + "Unsupported output data type");
  }
}

template <typename T>
void FillOutputHelper(Ort::CustomOpApi& ort, OrtValue* out_tensor, std::shared_ptr<ngraph::Node> node) {
  auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node);
  auto res = const_node->cast_vector<T>();
  T* tensor_data = ort.GetTensorMutableData<T>(out_tensor);
  std::copy(res.begin(), res.end(), tensor_data);
}

void FillInputBlob(InferenceEngine::Blob::Ptr& inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::CustomOpApi& ort, OrtKernelContext* context,
                   InferenceEngine::Precision precision, const SubGraphContext& subgraph_context) {
  auto minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputBlob);
  auto minputHolder = minput->wmap();

  auto input_data = minputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
  size_t input_data_size = inputBlob->byteSize();

  const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context.input_names.at(input_name));
  auto mem_info = ort.GetTensorMemoryInfo(tensor);
  if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
    ORT_THROW(log_tag + "IO Buffering is not enabled, Please enable Input on CPU");
  }
  auto tensor_shape = ort.GetTensorTypeAndShape(tensor);
  auto elem_type = ort.GetTensorElementType(tensor_shape);

  if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
      (precision == InferenceEngine::Precision::I32)) {
    const int64_t* tensor_data_64 = ort.GetTensorData<int64_t>(tensor);
    auto data_len = (input_data_size * 2) / sizeof(int64_t);
    const int64_t* batch_memory_offset = tensor_data_64 + data_len * batch_slice_idx;

    std::copy(batch_memory_offset, batch_memory_offset + data_len, (uint32_t*)input_data);
  } else {
    // Copy input data into OpenVINO's input buffer
    const char* tensor_data = ort.GetTensorData<char>(tensor);
    const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;
    std::memcpy(input_data, batch_memory_offset, input_data_size);
  }
}

void FillOutputBlob(InferenceEngine::Blob::Ptr& outputBlob, OrtValue* output_tensor,
                    Ort::CustomOpApi& ort, InferenceEngine::Precision precision, size_t batch_slice_idx) {
  auto moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob);

  auto moutputHolder = moutput->rmap();

  const auto output_data = moutputHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

  size_t output_data_size = outputBlob->byteSize();
  auto tensor_shape = ort.GetTensorTypeAndShape(output_tensor);
  auto elem_type = ort.GetTensorElementType(tensor_shape);

  if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
      (precision == InferenceEngine::Precision::I32)) {
    int64_t* tensor_data = ort.GetTensorMutableData<int64_t>(output_tensor);
    auto data_len = output_data_size / sizeof(int32_t);
    int64_t* batch_memory_offset = tensor_data + data_len * batch_slice_idx;

    std::transform((int32_t*)output_data, ((int32_t*)output_data) + data_len, batch_memory_offset, static_cast_int64());

  } else {
    char* tensor_data = ort.GetTensorMutableData<char>(output_tensor);
    char* batch_memory_offset = tensor_data + output_data_size * batch_slice_idx;

    std::memcpy(batch_memory_offset, output_data, output_data_size);
  }
}

std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>>
perfCountersSorted(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap) {
  using perfItem = std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>;
  std::vector<perfItem> sorted;
  for (auto& kvp : perfMap) sorted.push_back(kvp);

  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const perfItem& l, const perfItem& r) {
                     return l.second.execution_index < r.second.execution_index;
                   });

  return sorted;
}

#if defined (OV_API_20)
void FillInputBlob(OVTensorPtr inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::CustomOpApi& ort, OrtKernelContext* context,
                   const SubGraphContext& subgraph_context) {

    size_t input_data_size = inputBlob->get_byte_size();
    auto input_data = inputBlob->data();
    const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context.input_names.at(input_name));
    auto mem_info = ort.GetTensorMemoryInfo(tensor);
    if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
      ORT_THROW(log_tag + "IO Buffering is not enabled, Please enable Input on CPU");
    }
    // Copy input data into OpenVINO's input buffer
    const char* tensor_data = ort.GetTensorData<char>(tensor);
    const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;
    std::memcpy(input_data, batch_memory_offset, input_data_size);
}

void FillOutputBlob(OVTensorPtr outputBlob, OrtValue* output_tensor,
                    Ort::CustomOpApi& ort, size_t batch_slice_idx) {
  auto output_data = outputBlob->data();
  size_t output_data_size = outputBlob->get_byte_size();
  char* tensor_data = ort.GetTensorMutableData<char>(output_tensor);
  char* batch_memory_offset = tensor_data + output_data_size * batch_slice_idx;
  std::memcpy(batch_memory_offset, output_data, output_data_size);
}

void printPerformanceCounts(const std::vector<OVProfilingInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName) {
  long long totalTime = 0;
  // Print performance counts
  stream << std::endl
         << "performance counts:" << std::endl
         << std::endl;

  for (const auto& it : performanceMap) {
    std::string toPrint(it.node_name);
    const int maxLayerName = 30;

    if (it.node_name.length() >= maxLayerName) {
      toPrint = it.node_name.substr(0, maxLayerName - 4);
      toPrint += "...";
    }
    stream << std::setw(maxLayerName) << std::left << toPrint;
    switch (it.status) {
      case OVProfilingInfo::Status::EXECUTED:
        stream << std::setw(15) << std::left << "EXECUTED";
        break;
      case OVProfilingInfo::Status::NOT_RUN:
        stream << std::setw(15) << std::left << "NOT_RUN";
        break;
      case OVProfilingInfo::Status::OPTIMIZED_OUT:
        stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
        break;
    }
    stream << std::setw(30) << std::left << "layerType: " + std::string(it.node_type) + " ";
    stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.real_time.count());
    stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.cpu_time.count());
    stream << " execType: " << it.exec_type << std::endl;
    if (it.real_time.count() > 0) {
      totalTime += it.real_time.count();
    }
  }
  stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
  std::cout << std::endl;
  std::cout << "Full device name: " << deviceName << std::endl;
  std::cout << std::endl;
}
#endif

void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName) {
  long long totalTime = 0;
  // Print performance counts
  stream << std::endl
         << "performance counts:" << std::endl
         << std::endl;

  auto performanceMapSorted = perfCountersSorted(performanceMap);

  for (const auto& it : performanceMapSorted) {
    std::string toPrint(it.first);
    const int maxLayerName = 30;

    if (it.first.length() >= maxLayerName) {
      toPrint = it.first.substr(0, maxLayerName - 4);
      toPrint += "...";
    }
    stream << std::setw(maxLayerName) << std::left << toPrint;
    switch (it.second.status) {
      case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
        stream << std::setw(15) << std::left << "EXECUTED";
        break;
      case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
        stream << std::setw(15) << std::left << "NOT_RUN";
        break;
      case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
        stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
        break;
    }
    stream << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
    stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
    stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.second.cpu_uSec);
    stream << " execType: " << it.second.exec_type << std::endl;
    if (it.second.realTime_uSec > 0) {
      totalTime += it.second.realTime_uSec;
    }
  }
  stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
  std::cout << std::endl;
  std::cout << "Full device name: " << deviceName << std::endl;
  std::cout << std::endl;
}

void printPerformanceCounts(OVInferRequestPtr request, std::ostream& stream, std::string deviceName) {
  #if defined (OV_API_20)
    auto performanceMap = request->GetNewObj().get_profiling_info();
    printPerformanceCounts(performanceMap, stream, deviceName);
  #else
    auto performanceMap = request->GetObj().GetPerformanceCounts();
    printPerformanceCounts(performanceMap, stream, deviceName);
  #endif
}

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
