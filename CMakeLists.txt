cmake_minimum_required(VERSION 3.13.4)
project(mlir2crab LANGUAGES CXX C)

cmake_policy(SET CMP0116 OLD)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if (NOT CRAB_DIR)
  set(CRAB_DIR "" CACHE PATH "Root of Crab source tree") 
endif()
if (IS_DIRECTORY "${CRAB_DIR}" AND EXISTS ${CRAB_DIR}/CMakeLists.txt)
  message(STATUS "Using Crab in: ${CRAB_DIR}")
  set(CRAB_BINARY_DIR crab) ## needed if CRAB_DIR out of tree.
  add_subdirectory(${CRAB_DIR} ${CRAB_BINARY_DIR})
else()
  message(FATAL_ERROR "No Crab found in CRAB_DIR")
endif()

# Needed when included crab header files
find_package (Boost 1.65 REQUIRED)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_definitions(-DBOOST_NO_EXCEPTIONS)
else ()
  message(FATAL_ERROR "No boost found")
endif ()
  
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(LLVMDistributionSupport)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${CRAB_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/mlir2crab/Dialect)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)

