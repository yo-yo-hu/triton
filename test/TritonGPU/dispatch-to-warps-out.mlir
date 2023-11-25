#blocked = #triton_gpu.blocked<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [64, 32], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_0d1d2d3de4de5de6de7c8de9c10de11c(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = tt.get_warp_id : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = tt.get_program_id x : i32
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %1, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %1, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %1, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = arith.extsi %arg3 : i32 to i64
    %17 = arith.extsi %arg5 : i32 to i64
    %18 = arith.extsi %arg6 : i32 to i64
    %19 = arith.addi %0, %15 : i32
    %20 = arith.addi %0, %c0_i32 : i32
    %21 = tt.make_tensor_ptr %arg0, [%16, %17], [%18, %c1_i64], [%19, %20] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #blocked1>, 1>
    %22 = arith.muli %14, %c128_i32 : i32
    %23 = arith.extsi %arg4 : i32 to i64
    %24 = arith.extsi %arg7 : i32 to i64
    %25 = arith.addi %0, %c0_i32 : i32
    %26 = arith.addi %0, %22 : i32
    %27 = tt.make_tensor_ptr %arg1, [%17, %23], [%24, %c1_i64], [%25, %26] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #blocked2>, 1>
    %28:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %21, %arg12 = %27) -> (tensor<64x64xf32, #blocked>, !tt.ptr<tensor<32x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x32xf16, #blocked2>, 1>)  : i32 {
      %34 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #blocked1>, 1> -> tensor<32x32xf16, #blocked1>
      %35 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #blocked2>, 1> -> tensor<32x32xf16, #blocked2>
      %36 = triton_gpu.convert_layout %34 : (tensor<32x32xf16, #blocked1>) -> tensor<64x32xf16, #blocked3>
      %37 = triton_gpu.convert_layout %35 : (tensor<32x32xf16, #blocked2>) -> tensor<32x64xf16, #blocked4>
      %38 = tt.dot %36, %37, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf16, #blocked3> * tensor<32x64xf16, #blocked4> -> tensor<64x64xf32, #blocked>
      %39 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #blocked1>, 1>
      %40 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x32xf16, #blocked2>, 1>
      scf.yield %38, %39, %40 : tensor<64x64xf32, #blocked>, !tt.ptr<tensor<32x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x32xf16, #blocked2>, 1>
    }
    %29 = arith.truncf %28#0 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %30 = arith.extsi %arg8 : i32 to i64
    %31 = arith.addi %0, %15 : i32
    %32 = arith.addi %0, %22 : i32
    %33 = tt.make_tensor_ptr %arg2, [%16, %23], [%30, %c1_i64], [%31, %32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #blocked>, 1>
    tt.store %33, %29 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<64x64xf16, #blocked>
    tt.return
  }
}

