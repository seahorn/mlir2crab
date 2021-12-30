get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
	MLIR2Crab
        )
add_llvm_executable(mlir2crab mlir2crab.cpp)

llvm_update_compile_flags(mlir2crab)
target_link_libraries(mlir2crab PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir2crab)

install(TARGETS mlir2crab DESTINATION bin)
