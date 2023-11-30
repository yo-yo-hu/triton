#blocked = #triton_gpu.blocked<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_with_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = tt.get_warp_id : i32
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
    %19 = arith.muli %0, %c32_i32 : i32
    %20 = arith.addi %19, %15 : i32
    %21 = tt.make_tensor_ptr %arg0, [%16, %17], [%18, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #blocked1>, 1>
    %22 = arith.muli %14, %c128_i32 : i32
    %23 = arith.extsi %arg4 : i32 to i64
    %24 = arith.extsi %arg7 : i32 to i64
    %25 = arith.addi %19, %22 : i32
    %26 = tt.make_tensor_ptr %arg1, [%17, %23], [%24, %c1_i64], [%c0_i32, %25] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #blocked2>, 1>
    %27:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %21, %arg12 = %26) -> (tensor<64x64xf32, #blocked>, !tt.ptr<tensor<32x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x32xf16, #blocked2>, 1>)  : i32 {
      %39 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #blocked1>, 1> -> tensor<32x32xf16, #blocked1>
      %40 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #blocked2>, 1> -> tensor<32x32xf16, #blocked2>
      %41 = triton_gpu.alloc : <f16, 1>
      %42 = tt.make_tensor_ptr %41, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%19, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #blocked1>, 3>
      tt.store %42, %39 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x32xf16, #blocked1>, 3>, tensor<32x32xf16, #blocked1>
      gpu.barrier
      %43 = arith.divsi %0, %c2_i32 : i32
      %44 = arith.remsi %43, %c2_i32 : i32
      %45 = arith.muli %44, %c64_i32 : i32
      %46 = tt.make_tensor_ptr %41, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%45, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 3>
      %47 = tt.load %46 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 3> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %48 = triton_gpu.alloc : <f16, 1>
      %49 = tt.make_tensor_ptr %48, [%c32_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #blocked2>, 3>
      tt.store %49, %40 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x32xf16, #blocked2>, 3>, tensor<32x32xf16, #blocked2>
      gpu.barrier
      %50 = arith.remsi %0, %c2_i32 : i32
      %51 = arith.remsi %50, %c2_i32 : i32
      %52 = arith.muli %51, %c64_i32 : i32
      %53 = tt.make_tensor_ptr %48, [%c32_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %52] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 3>
      %54 = tt.load %53 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 3> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %55 = tt.dot %47, %54, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %56 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #blocked1>, 1>
      %57 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x32xf16, #blocked2>, 1>
      scf.yield %55, %56, %57 : tensor<64x64xf32, #blocked>, !tt.ptr<tensor<32x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x32xf16, #blocked2>, 1>
    }
    %28 = arith.truncf %27#0 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %29 = arith.extsi %arg8 : i32 to i64
    %30 = arith.divsi %0, %c2_i32 : i32
    %31 = arith.remsi %30, %c2_i32 : i32
    %32 = arith.muli %31, %c64_i32 : i32
    %33 = arith.addi %32, %15 : i32
    %34 = arith.remsi %0, %c2_i32 : i32
    %35 = arith.remsi %34, %c2_i32 : i32
    %36 = arith.muli %35, %c64_i32 : i32
    %37 = arith.addi %36, %22 : i32
    %38 = tt.make_tensor_ptr %arg2, [%16, %23], [%29, %c1_i64], [%33, %37] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #blocked>, 1>
    tt.store %38, %28 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<64x64xf16, #blocked>
    tt.return
  }
  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = tt.get_warp_id : i32
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
    %19 = arith.divsi %0, %c2_i32 : i32
    %20 = arith.remsi %19, %c2_i32 : i32
    %21 = arith.muli %20, %c64_i32 : i32
    %22 = arith.addi %21, %15 : i32
    %23 = tt.make_tensor_ptr %arg0, [%16, %17], [%18, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 1>
    %24 = arith.muli %14, %c128_i32 : i32
    %25 = arith.extsi %arg4 : i32 to i64
    %26 = arith.extsi %arg7 : i32 to i64
    %27 = arith.remsi %0, %c2_i32 : i32
    %28 = arith.remsi %27, %c2_i32 : i32
    %29 = arith.muli %28, %c64_i32 : i32
    %30 = arith.addi %29, %24 : i32
    %31 = tt.make_tensor_ptr %arg1, [%17, %25], [%26, %c1_i64], [%c0_i32, %30] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 1>
    %32:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %23, %arg12 = %31) -> (tensor<64x64xf32, #blocked>, !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 1>, !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 1>)  : i32 {
      %36 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 1> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %37 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 1> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %38 = tt.dot %36, %37, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %39 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 1>
      %40 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 1>
      scf.yield %38, %39, %40 : tensor<64x64xf32, #blocked>, !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, 1>, !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, 1>
    }
    %33 = arith.truncf %32#0 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %34 = arith.extsi %arg8 : i32 to i64
    %35 = tt.make_tensor_ptr %arg2, [%16, %25], [%34, %c1_i64], [%22, %30] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #blocked>, 1>
    tt.store %35, %33 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<64x64xf16, #blocked>
    tt.return
  }
}

