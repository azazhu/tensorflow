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
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace disc_ral {

using LLVM::LLVMFuncOp;
static constexpr const char* kMalloc = "alloc";
constexpr const char* kRalDispatchFunctionName = "disc_ral_call";

namespace {

// A rewrite pattern to convert memref.alloc operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
class ConvertMemRefAllocOpToDispatchOpPattern
    : public ConvertOpToLLVMPattern<memref::AllocOp> {
 public:
  ConvertMemRefAllocOpToDispatchOpPattern(LLVMTypeConverter& type_converter)
      : ConvertOpToLLVMPattern<memref::AllocOp>(type_converter) {}

 private:
  LogicalResult matchAndRewrite(
      memref::AllocOp alloc_op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override;
};

// Emits LLVM IR to malloc a device memory. 
LogicalResult ConvertMemRefAllocOpToDispatchOpPattern::matchAndRewrite(
    memref::AllocOp alloc_op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const {
  mlir::Operation* op = alloc_op.getOperation();
  ModuleOp module = op->getParentOfType<ModuleOp>();

  Location loc = op->getLoc();
  auto memref = alloc_op.getResult();
  MemRefType memref_type = memref.getType().cast<MemRefType>();
  Attribute memorySpace = memref_type.getMemorySpace();
  if (!memorySpace) {
    return failure();
  }
  if (!memorySpace.isa<StringAttr>()) {
    return failure();
  }

  // TODO(disc) : handle LLVM::LLVMFuncOp
  FuncOp func_op = alloc_op->getParentOfType<FuncOp>();
  if (!func_op) {
    return failure();
  }
  Value context_arg = func_op.getArgument(0);

  Type llvm_pointer_type =
      LLVM::LLVMPointerType::get(IntegerType::get(alloc_op->getContext(), 8));

  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  auto new_memref_ty =
      MemRefType::get(memref_type.getShape(), memref_type.getElementType(),
                      memref_type.getAffineMaps());

  Type llvm_int8_type = IntegerType::get(rewriter.getContext(), 8);

  // Set all dynamic sizes to 1 and compute fake strides.
  SmallVector<Value, 4> dyn_sizes(memref_type.getNumDynamicDims(),
                                  createIndexConstant(rewriter, loc, 1));
  // Get memref descriptor sizes.
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Value sizeBytes;

  getMemRefDescriptorSizes(loc, memref_type, dyn_sizes, rewriter, sizes,
                           strides, sizeBytes);

  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      alloc_op, memref_type, context_arg, sizeBytes, kMalloc, false, "device");

  return success();
}

class ConvertAddressSpacePass
    : public ConvertAddressSpacePassBase<ConvertAddressSpacePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect, disc_ral::RalDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext* ctx = m.getContext();
    LLVMTypeConverter type_converter(m.getContext());
    type_converter.addConversion(
        [&](mlir::disc_ral::RalExecutionContextType type) {
          return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
        });

    // Populate patterns.
    RewritePatternSet patterns(&getContext());

    populateStdExpandOpsPatterns(patterns);
    populateStdToLLVMConversionPatterns(type_converter, patterns);
    populateComplexToLLVMConversionPatterns(type_converter, patterns);
    populateVectorToLLVMConversionPatterns(type_converter, patterns);
    populateMathToLibmConversionPatterns(patterns, 0);
    PopulateMemRefDeviceAllocToLLVMConversionPatterns(&type_converter,
                                                      &patterns);

    // Set target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<disc_ral::RalDialect>();
    // Mark modules as legal.
    target.addLegalOp<ModuleOp, gpu::GPUModuleOp>();
    target.addLegalOp<FuncOp>();
    // target.addLegalOp<disc_ral::DispatchOp>();
    // TODO(disc): set Illegal
    target.addLegalDialect<StandardOpsDialect>();

    target.addIllegalDialect<complex::ComplexDialect, math::MathDialect>();
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

void PopulateMemRefDeviceAllocToLLVMConversionPatterns(
    LLVMTypeConverter* converter, RewritePatternSet* patterns) {
  // clang-format off
  patterns->insert<
      ConvertMemRefAllocOpToDispatchOpPattern
    >(*converter);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp> > createConvertAddressSpacePass() {
  return std::make_unique<ConvertAddressSpacePass>();
}

}  // namespace disc_ral
}  // namespace mlir
