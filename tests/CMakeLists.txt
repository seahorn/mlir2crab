# find lit. Assume we are using external lit installed from pip
#find_program(LIT_COMMAND lit REQUIRED)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
      
add_lit_testsuite(check-mlir2crab "Run simple tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS mlir2crab
)

set_target_properties(check-mlir2crab PROPERTIES FOLDER "Tests")

