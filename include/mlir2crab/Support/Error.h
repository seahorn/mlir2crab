#pragma once

#include "mlir2crab/Support/Log.h"
#include <cstdlib>

namespace mlir2crab {
#define ERROR_AND_ABORT(msg) \
  ERR << msg;		     \
  std::exit(EXIT_FAILURE);
} // end namespace mlir2crab
