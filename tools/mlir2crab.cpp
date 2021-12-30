//===- mlir2crab-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir2crab/Dialect/Crab/IR/CrabDialect.h"
#include "mlir2crab/mlir2crab.h"
#include "mlir2crab/CrabIrBuilderOpts.h"
#include "mlir2crab/Support/Log.h"

using namespace mlir;
using namespace llvm;

static llvm::cl::opt<std::string>
InputFilename(llvm::cl::Positional,
	      llvm::cl::desc("<input file>"),
	      llvm::cl::Required, llvm::cl::value_desc("filename"));
  
//static llvm::cl::opt<std::string>
//OutputFilename("o", llvm::cl::desc("Output filename"),
//	       llvm::cl::init(""), llvm::cl::value_desc("filename"));

// Prove assertions
static llvm::cl::opt<bool>
Verify("verify", 
       llvm::cl::desc("Run Crab Assertion Checker"),
       llvm::cl::init(true));

// Choose abstract domain
static llvm::cl::opt<mlir2crab::Domain>
AbsDomain("abstract-domain",
   llvm::cl::desc("Choose abstract domain"),
   llvm::cl::values
	  (clEnumValN(mlir2crab::Domain::Intervals, "intervals", "Classical interval domain"),
	   clEnumValN(mlir2crab::Domain::Zones    , "zones", "Zones domain (aka DBM)"),
	   clEnumValN(mlir2crab::Domain::Octagons , "octagons", "Octagons domain")),
	  llvm::cl::init(mlir2crab::Domain::Zones));  

int main(int argc, char **argv) {
    // Register our used dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::crab::CrabDialect>();
    //registry.insert<mlir::StandardOpsDialect>();

    // Set up needed tools
    InitLLVM y(argc, argv);
    cl::ParseCommandLineOptions(argc, argv);

    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(InputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return EXIT_FAILURE;
    }

    // auto output = openOutputFile(OutputFilename, &errorMessage);
    // if (!output) {
    //     llvm::errs() << errorMessage << "\n";
    //     return EXIT_FAILURE;
    // }

    // Tell sourceMgr about this buffer; parser will pick this up
    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    // New context for our buffer
    MLIRContext context(registry);
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    OwningModuleRef module(parseSourceFile(sourceMgr, &context));        
    if (!module) {
      llvm::errs() << "Module is a null pointer\n";
      return EXIT_FAILURE;
    }

    // Generate CrabIR
    mlir2crab::CrabIrBuilderOpts builder_opts;
    mlir2crab::CrabIrBuilder crabIrBuilder(std::move(module), builder_opts);
    crabIrBuilder.generate();

    // Analyze CrabIR
    mlir2crab::CrabIrAnalyzerOpts analyzer_opts;
    analyzer_opts.run_checker = Verify;
    analyzer_opts.domain = AbsDomain;
    mlir2crab::CrabIrAnalyzer crabIrAnalyzer(crabIrBuilder, analyzer_opts);
    crabIrAnalyzer.analyze();
    llvm::outs() << crabIrAnalyzer << "\n";
    
    return EXIT_SUCCESS;
}
