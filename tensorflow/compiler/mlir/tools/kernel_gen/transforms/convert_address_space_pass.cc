/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdexcept>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"      // from @llvm-project
#include "mlir/Conversion/MathToLibm/MathToLibm.h"        // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"           // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"            // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"                // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"            // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"              // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"         // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"                  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"                        // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"                 // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"        // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"             // from @llvm-project
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/Passes.h"       // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                          // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"           // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"


namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"


class ConvertAddressSpacePass
    : public ConvertAddressSpacePassBase<ConvertAddressSpacePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext* ctx = m.getContext();
    LLVMTypeConverter type_converter(m.getContext());
    type_converter.addConversion([&](RalExecutionContextType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    });
    
    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    
    populateStdExpandOpsPatterns(patterns);
    populateStdToLLVMConversionPatterns(type_converter, patterns);
    populateComplexToLLVMConversionPatterns(type_converter, patterns);
    populateVectorToLLVMConversionPatterns(type_converter, patterns);
    populateMathToLibmConversionPatterns(patterns, 0);
    transforms::PopulateMemRefDeviceAllocToLLVMConversionPatterns(&type_converter, &patterns);

    // Set target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<disc_ral::RalDialect>();
    // Mark modules as legal.
    target.addLegalOp<ModuleOp, gpu::GPUModuleOp>();
    target.addLegalOp<FuncOp>();
    // TODO(disc): set Illegal
    target.addLegalDialect<StandardOpsDialect>();
    
    target.addIllegalDialect<complex::ComplexDialect,
                             gpu::GPUDialect, 
                             math::MathDialect>();
    // TODO(disc): uncomment
    // target.addIllegalOp<LLVM::DialectCastOp>();
    target.addIllegalOp<memref::AllocOp>();

    // Do not look into gpu modules, only consider host-side.
    // target.markOpRecursivelyLegal<gpu::GPUModuleOp>();

    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
    // Finally, strip the GPU modules, as they are no longer needed.
    for (auto op : llvm::make_early_inc_range(m.getOps<gpu::GPUModuleOp>())) {
      op.erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateConvertAddressSpacePass() {
  return std::make_unique<ConvertAddressSpacePass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
