// tf-opt --convert-std-to-llvm  -canonicalize tensorflow/compiler/mlir/hlo/tests/std_to_llvm_convert_address_space.mlir
module  {
  func @main(%arg0: !disc_ral.context) {
    %c1024 = constant 1024 : index
    %c1024_0 = constant 1024 : index
    %0 = memref.alloc() : memref<2x3x4xf32, "device_type_3df32">
    %1 = memref.alloc() : memref<2x3x4xf32, "device_type_3df32">
    %2 = memref.alloc() : memref<2x3x4xf32, "device_type_3df32">
    %3 = call @non_fusion_elemwise_cpu(%0, %1, %2) : (memref<2x3x4xf32, "device_type_3df32">, memref<2x3x4xf32, "device_type_3df32">, memref<2x3x4xf32, "device_type_3df32">) -> memref<2x3x4xf32, "device_type_3df32">
    return
  }

  func @non_fusion_elemwise_cpu(%arg0: memref<2x3x4xf32, "device_type_3df32">, %arg1: memref<2x3x4xf32, "device_type_3df32">, %arg2: memref<2x3x4xf32, "device_type_3df32">) -> memref<2x3x4xf32, "device_type_3df32"> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg2, %c0 : memref<2x3x4xf32, "device_type_3df32">
    %1 = memref.dim %arg2, %c1 : memref<2x3x4xf32, "device_type_3df32">
    %2 = muli %0, %1 : index
    %3 = memref.dim %arg2, %c2 : memref<2x3x4xf32, "device_type_3df32">
    %4 = muli %2, %3 : index
    scf.for %arg3 = %c0 to %4 step %c1 {
      %5 = muli %3, %1 : index
      %6 = divi_unsigned %arg3, %5 : index
      %7 = remi_unsigned %arg3, %5 : index
      %8 = divi_unsigned %7, %3 : index
      %9 = remi_unsigned %7, %3 : index
      %10 = memref.load %arg0[%6, %8, %9] : memref<2x3x4xf32, "device_type_3df32">
      %11 = memref.load %arg1[%6, %8, %9] : memref<2x3x4xf32, "device_type_3df32">
      %12 = addf %10, %11 : f32
      %13 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<2x3x4xf32, "device_type_3df32"> to memref<?xf32, "device_type_3df32">
      memref.store %12, %13[%arg3] : memref<?xf32, "device_type_3df32">
    }
    return %arg2 : memref<2x3x4xf32, "device_type_3df32">
  }
}
