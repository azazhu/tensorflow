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
#include <string>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"    // from @llvm-project
#include "mlir/IR/Attributes.h"                 // from @llvm-project
#include "mlir/IR/BuiltinOps.h"                 // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"               // from @llvm-project
#include "mlir/IR/Operation.h"                  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {

namespace {

using LLVM::LLVMFuncOp;
static constexpr const char* kMalloc = "malloc";
constexpr const char* kRalDispatchFunctionName = "disc_ral_call";

// A rewrite pattern to convert gpu.launch_func operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
class ConvertMemRefOpToRalCallPattern
    : public ConvertOpToLLVMPattern<memref::AllocOp> {
 public:
  ConvertMemRefOpToRalCallPattern(LLVMTypeConverter& type_converter)
      : ConvertOpToLLVMPattern<memref::AllocOp>(type_converter) {}

 private:
  LogicalResult matchAndRewrite(
      memref::AllocOp alloc_op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override;
};

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the `kRalGpuLaunch` attribute.
LogicalResult ConvertMemRefOpToRalCallPattern::matchAndRewrite(
    memref::AllocOp alloc_op, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const {
  llvm::dbgs() << "MemRefAllocOpConverter matchAndRewrite\n";
  mlir::Operation* op = alloc_op.getOperation();
  ModuleOp module = op->getParentOfType<ModuleOp>();

  Location loc = op->getLoc();
  auto memref = alloc_op.getResult();
  llvm::dbgs() << "memref " << memref << "\n";
  MemRefType memref_type = memref.getType().cast<MemRefType>();
  Attribute memorySpace = memref_type.getMemorySpace();
  if (!memorySpace) {
    return failure();
  }
  if (!memorySpace.isa<StringAttr>()) {
    return failure();
  }
  llvm::dbgs() << "here0\n";
  Value context_arg =
      alloc_op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
      
  llvm::dbgs() << "here1\n";
  Type llvm_pointer_type =
      LLVM::LLVMPointerType::get(IntegerType::get(alloc_op->getContext(), 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  llvm::dbgs() << "here2\n";
  auto new_memref_ty =
      MemRefType::get(memref_type.getShape(), memref_type.getElementType(),
                      memref_type.getAffineMaps());
  llvm::dbgs() << "here3\n";
  Type llvm_int32_type = IntegerType::get(rewriter.getContext(), 32);
  llvm::dbgs() << "here4\n";
  Value c_1024 = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, rewriter.getI32IntegerAttr(1024));
  llvm::dbgs() << "c_1024 " << c_1024 << "\n";
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      alloc_op, llvm::None, context_arg, c_1024, kMalloc, false, "cpu");
  return success();
}

class MemRefAllocOpConverter : public ConvertOpToLLVMPattern<memref::AllocOp> {
 public:
  using ConvertOpToLLVMPattern<memref::AllocOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp alloc_op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    llvm::dbgs() << "MemRefAllocOpConverter matchAndRewrite\n";
    mlir::Operation* op = alloc_op.getOperation();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Location loc = op->getLoc();
    auto memref = alloc_op.getResult();
    llvm::dbgs() << "memref " << memref << "\n";
    MemRefType memref_type = memref.getType().cast<MemRefType>();
    Attribute memorySpace = memref_type.getMemorySpace();
    if (!memorySpace) {
      return failure();
    }
    if (!memorySpace.isa<StringAttr>()) {
      return failure();
    }
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());

    Type llvm_pointer_type =
        LLVM::LLVMPointerType::get(IntegerType::get(alloc_op->getContext(), 8));
    Type llvm_pointer_pointer_type =
        LLVM::LLVMPointerType::get(llvm_pointer_type);

    std::string mem_space_str = memorySpace.cast<StringAttr>().getValue().str();

    auto new_memref_ty =
        MemRefType::get(memref_type.getShape(), memref_type.getElementType(),
                        memref_type.getAffineMaps());

    // Allocate memory for the coroutine frame.
    LLVMFuncOp dispatch_func = rewriter.create<LLVMFuncOp>(
        op->getLoc(), kRalDispatchFunctionName,
        LLVM::LLVMFunctionType::get(
            getVoidType(),
            {
                llvm_pointer_type,        /* ral_context_t */
                llvm_pointer_type,        /* void* call_target_name */
                llvm_pointer_pointer_type /* void** args */
            },
            /*isVarArg=*/false));
    rewriter.restoreInsertionPoint(ip);

    auto coroSize =
        rewriter.create<LLVM::CoroSizeOp>(loc, rewriter.getI64Type());

    auto coroAlloc = rewriter.create<LLVM::CallOp>(
        loc, llvm::None, rewriter.getSymbolRefAttr(dispatch_func),
        ValueRange(coroSize.getResult()));
    llvm::dbgs() << "here2 " << coroAlloc << "\n";
    rewriter.replaceOp(alloc_op, coroAlloc->getResults());
    llvm::dbgs() << "here3\n";
    /*
    rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(alloc_op,
    memref.getType(), operands.front(), operands.drop_front(), "ral_alloc",
                                            false, "cpu");
    */
    // MemRefType memref_type = alloc_op.getType();
    return success();
  }
};

}  // namespace

void PopulateMemRefDeviceAllocToLLVMConversionPatterns(
    LLVMTypeConverter* converter, RewritePatternSet* patterns) {
  llvm::dbgs() << "PopulateMemRefDeviceAllocToLLVMConversionPatterns\n";
  // clang-format off
  patterns->insert<
      ConvertMemRefOpToRalCallPattern
    >(*converter);
  // clang-format on
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir