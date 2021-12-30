#include "mlir2crab/CrabIrBuilderOpts.h"

namespace mlir2crab {
void CrabIrBuilderOpts::write(llvm::raw_ostream &o) const {
  o << "=== Crab builder options === \n";
  o << "No options\n";
}
} //end namespace mlir2crab
