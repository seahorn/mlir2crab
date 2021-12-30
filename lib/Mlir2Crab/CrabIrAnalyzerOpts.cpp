#include "mlir2crab/CrabIrAnalyzerOpts.h"

namespace mlir2crab {
void CrabIrAnalyzerOpts::write(llvm::raw_ostream &o) const {
  o << "== Crab analyzer options == \n";
  o << "Abstract domain: ";
  switch(domain) {
  case Domain::Intervals:
    o << "Intervals";
    break;
  case Domain::Zones:
    o << "Zones";
    break;
  default:
    o << "Octagons";
    break;
  }
  o << "\n";
  o << "Run checker: " << run_checker << "\n";    
}
} //end namespace mlir2crab
