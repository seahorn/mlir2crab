// RUN: %mlir2crab %s  | FileCheck %s
// CHECK: 1  Number of total safe checks
// CHECK: 0  Number of total error checks
// CHECK: 0  Number of total warning checks

module  { 
  func @main() -> i32 {
    %0 = crab.const 0 : i32
    %1 = crab.const 0 : i32
    crab.br ^bb1(%0, %1 : i32, i32)
  ^bb1(%2: i32, %3: i32):  // 2 preds: ^bb0, ^bb2
    crab.nd_br ^bb2(%2, %3 : i32, i32), ^bb3(%2, %3 : i32, i32)
  ^bb2(%5: i32, %6: i32):  // pred: ^bb1
    %7 = crab.const 9 : i32
    crab.assume sle(%5, %7) : i32
    %8 = crab.const 1 : i32
    %9 = crab.add(%5, %8) : i32
    %10 = crab.add(%6, %8) : i32
    crab.br ^bb1(%9, %10 : i32, i32)
  ^bb3(%11: i32, %12: i32):  // pred: ^bb1
    %13 = crab.const 10 : i32
    crab.assume sge(%11, %13) : i32
    crab.br ^bb4(%11, %12 : i32, i32)
  ^bb4(%14: i32, %15: i32):  // pred: ^bb3
    crab.assert eq(%14, %15) : i32
    %16 = crab.const 0 : i32
    crab.return %16 : i32
  }
}
