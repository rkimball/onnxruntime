// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <map>
#include <utility>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/platform/env.h"
#include "core/graph/model.h"

#include "tvm_execution_provider.h"
#include "xpu_data_transfer.h"
#include "tvm_allocator.h"
#include "tvm_utils.h"
#include "tvm_api.h"


using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace tvm {

TvmExecutionProvider::TvmExecutionProvider(const TvmEPOptions& options)
    : IExecutionProvider{kTvmExecutionProvider},
      options_{options} {
}

TvmExecutionProvider::~TvmExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
TvmExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  return result;
}

void TvmExecutionProvider::printOptions() {
  LOGS(*GetLogger(), INFO) << options_;
}

void TvmExecutionProvider::setInputShapesForFreezedNN(const GraphViewer& graph_viewer,
                                                      TVMTensorShapes& input_shapes,
                                                      InputsInfoMap& all_input_shapes) {
  const std::vector<const NodeArg*>& all_nodes = graph_viewer.GetInputsIncludingInitializers();

  size_t indx = 0;
  for (const auto* node : all_nodes) {
    if (!graph_viewer.IsInitializedTensor(node->Name())) {
      TensorShapeVector shape = getInputShape(node);
      all_input_shapes[indx++] = shape;
      input_shapes.emplace_back(shape);
    }
  }
}

void TvmExecutionProvider::setInputShapesForUnfreezedNN(const GraphViewer& graph_viewer,
                                                        TVMTensorShapes& input_shapes,
                                                        InputsInfoMap& all_input_shapes) {
  const std::vector<const NodeArg*>& all_nodes = graph_viewer.GetInputsIncludingInitializers();

  size_t indx = 0;
  for (const auto* node : all_nodes) {
    TensorShapeVector shape = getInputShape(node);
    all_input_shapes[indx++] = shape;
    if (!graph_viewer.IsInitializedTensor(node->Name())) {
      input_shapes.emplace_back(shape);
    }
  }
}

TensorShapeVector TvmExecutionProvider::getInputShape(const NodeArg* node) {
    TensorShapeVector shape;
    const auto& node_name = node->Name();
    if(!options_.input_shapes.empty() &&
        options_.input_shapes.count(node_name)) {
      shape = options_.input_shapes[node_name];
    } else {
      shape = convertTensorShape(*node->Shape());
    }

    return shape;
}

TensorShapeVector TvmExecutionProvider::convertTensorShape(const TensorShapeProto& shape_proto) {
  TensorShape ort_shape = utils::GetTensorShapeFromTensorShapeProto(shape_proto);
  size_t dims = ort_shape.NumDimensions();

  TensorShapeVector shape(dims);
  for (size_t j = 0; j < dims; ++j) {
    int64_t dim = int64_t(ort_shape[j]);
    ORT_ENFORCE(dim > 0, "Input dimension is not positive value (dim = " + std::to_string(dim) + "). " +
      "Please use provider options to setup input_names and input_shapes");
    shape[j] = dim;
  }

  return shape;
}

void TvmExecutionProvider::prepareOutputTensors(const std::shared_ptr<TvmModule>& mod,
                                                std::vector<DLTensor>& output_tensors,
                                                size_t num) {
  ORT_ENFORCE(mod != nullptr, "TVM module is not compiled");
  output_tensors.clear();
  options_.output_shapes.clear();
  options_.output_shapes.resize(num);

  if (options_.executor != "vm") {
    TVMGetOutputShapes(*mod, options_.output_shapes);
  }

  for (auto& output_shape : options_.output_shapes) {
    DLTensor t;
    // Draft for tensor, correct data is defined during inference
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = nullptr;
    if (options_.executor == "vm") {
      t.ndim = 0;
      t.shape = nullptr;
    } else {
      t.ndim = output_shape.size();
      t.shape = output_shape.data();
    }

    output_tensors.push_back(t);
  }
}

}  // namespace tvm
}  // namespace onnxruntime
