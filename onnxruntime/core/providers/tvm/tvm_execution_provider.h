// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_EXECUTION_PROVIDER_H
#define TVM_EXECUTION_PROVIDER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"

#include "tvm_compiler.h"
#include "tvm_runner.h"


namespace onnxruntime {
  class Graph;
  class NodeArg;
namespace tvm {

class TvmExecutionProvider : public IExecutionProvider {
  using Compiler = TVMCompilerBase;
  using Compilers = std::unordered_map<std::string, std::shared_ptr<Compiler>>;
  using Runner = TVMRunner;
  using Runners = std::unordered_map<std::string, std::shared_ptr<Runner>>;

 public:
  explicit TvmExecutionProvider(const TvmEPOptions& options);
  virtual ~TvmExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;

 private:
  TvmEPOptions options_;
};

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_EXECUTION_PROVIDER_H
