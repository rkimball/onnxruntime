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
                                    const IKernelLookup&) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  return result;
}

}  // namespace tvm
}  // namespace onnxruntime
