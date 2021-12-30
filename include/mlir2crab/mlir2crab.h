#pragma once

#include "mlir2crab/CrabIrBuilderOpts.h"
#include "mlir2crab/CrabIrAnalyzerOpts.h"
#include "mlir2crab/CrabIrTypes.h"
#include <memory>

namespace llvm {
class raw_ostream;
}
namespace mlir {
class OwningModuleRef;
} // end namespace mlir

namespace mlir2crab {
class CrabIrBuilderImpl;
class CrabIrAnalyzerImpl;
} // end namespace mlir2crab

namespace mlir2crab {
class CrabIrBuilder {
  std::unique_ptr<CrabIrBuilderImpl> m_impl;
public:    
  CrabIrBuilder(mlir::OwningModuleRef &&module, const CrabIrBuilderOpts &opts);
  CrabIrBuilder(const CrabIrBuilder &other) = delete;
  CrabIrBuilder& operator==(const CrabIrBuilder &other) = delete;  
  ~CrabIrBuilder();

  const CrabIrBuilderOpts& getOpts() const;  
  // Translate module to CrabIR
  void generate();
  // Return a Crab callgraph that models the semantics of module
  // Precondition: generate() has been called.
  const callgraph_t &getCallGraph() const;
  callgraph_t &getCallGraph();
};

class CrabIrAnalyzer {
  std::unique_ptr<CrabIrAnalyzerImpl> m_impl;
public:
  CrabIrAnalyzer(CrabIrBuilder &crabIR, const CrabIrAnalyzerOpts &opts);
  CrabIrAnalyzer(const CrabIrAnalyzer &other) = delete;
  CrabIrAnalyzer& operator==(const CrabIrAnalyzer &other) = delete;
  ~CrabIrAnalyzer();

  const CrabIrAnalyzerOpts& getOpts() const;
  void analyze();
  void write(llvm::raw_ostream &os) const;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream &o,
				       const CrabIrAnalyzer &analyzer);

  // TODO: Add API to extract invariants
};  
} // end namespace mlir2crab
