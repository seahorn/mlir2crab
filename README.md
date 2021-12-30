# A MLIR front-end for Crab

Based on [mlir example](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone). Adapted compilation instructions. Disabled tests.

## Building LLVM
Commands to configure and compile LLVM

```sh
$ mkdir debug && cd debug 
$ cmake -G Ninja ../llvm \
    -DCMAKE_C_COMPILER=clang-12 -DCMAKE_CXX_COMPILER=clang++-12 \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON  \ 
    -DCMAKE_BUILD_TYPE=Debug \ # change to RelWithDebInfo for release build
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"  \ # maybe only X86 is needed
    -DLLVM_ENABLE_LLD=ON  \ # only on Linux	
    -DLLVM_INSTALL_UTILS=ON \ # optional to install FileCheck and lit
    -DCMAKE_INSTALL_PREFIX=<LLVM_PROJECT_INSTALL_DIR>
$ ninja
$ ninja install
```

The above installs a debug version of llvm in
`<LLVM_PROJECT_INSTALL_DIR>`.

## Download Crab


```sh
export CRAB_DIR=$(pwd)/crab
git clone git@github.com:seahorn/crab.git
```

## Building
To compile this project

```sh
$ mkdir debug && cd debug 
$ cmake -G Ninja .. \
    -DMLIR_DIR=${LLVM_PROJECT_INSTALL_DIR}/lib/cmake/mlir \
    -DLLVM_DIR=${LLVM_PROJECT_INSTALL_DIR}/lib/cmake/llvm \
	-DCRAB_DIR=${CRAB_DIR}  \	
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> 
$ ninja install	
```

The above assumes that `lit` command is installed and is globally
available. The above installs the binaries and libraries of this
project in `<INSTALL_DIR>`.

## Usage

``` sh
$ <INSTALL_DIR>/bin/mlir2crab test.mlir

```


## Contributors
Arie Gurfinkel <arie.gurfinkel@uwaterloo.ca> 
Joseph Tafese <jetafese@uwaterloo.ca> 
Jorge Navas <navasjorgea@gmail.com>
