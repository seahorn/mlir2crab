add_llvm_library(MLIR2Crab DISABLE_LLVM_LINK_LLVM_DYLIB
  CrabIrBuilder.cpp
  CrabIrAnalyzer.cpp
  CrabIrBuilderOpts.cpp
  CrabIrAnalyzerOpts.cpp
  Log.cpp
  BoostException.cpp
  )

llvm_map_components_to_libnames(LLVM_LIBS
  support core irreader bitwriter instrumentation ipo scalaropts codegen objcarcopts)

target_link_libraries (MLIR2Crab
  PRIVATE
  ${CRAB_LIBS}
  ${LLVM_LIBS})

install(TARGETS MLIR2Crab
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
