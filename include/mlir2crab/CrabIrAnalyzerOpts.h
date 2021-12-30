#pragma once

#include "llvm/Support/raw_ostream.h"

namespace mlir2crab {

enum class Domain {
  Intervals,
  Zones,
  Octagons
};
  
struct CrabIrAnalyzerOpts {
  Domain domain;
  bool run_checker;
  
  CrabIrAnalyzerOpts()
    : domain(Domain::Zones),
      run_checker(true) {
  }
    
  void write(llvm::raw_ostream &o) const;  
};
} // end namepace mlir2crab
