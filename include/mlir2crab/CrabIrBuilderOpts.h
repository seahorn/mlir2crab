#pragma once

#include "llvm/Support/raw_ostream.h"

namespace mlir2crab {
struct CrabIrBuilderOpts {  
  // add here all options
  CrabIrBuilderOpts() {}
  void write(llvm::raw_ostream &o) const;  
};
} //end namespace mlir2crab

