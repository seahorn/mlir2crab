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
## Example

``` sh
$ mlir2crab tests/p1.mlir
```

```
void declare main()
^bb0:
  %0 = 0;
  %1 = 0;
  goto ^bb1;
^bb1:
  goto edge_^bb1_^bb2,edge_^bb1_^bb3;
edge_^bb1_^bb2:
  %2 = %0;
  %3 = %1;
  goto ^bb2;
^bb2:
  assume(%2 <= 9);
  %6 = %2+1;
  %7 = %3+1;
  %0 = %6;
  %1 = %7;
  goto ^bb1;
edge_^bb1_^bb3:
  %4 = %0;
  %5 = %1;
  goto ^bb3;
^bb3:
  assume(-%4 <= -10);
  %8 = %4;
  %9 = %5;
  goto ^bb4;
^bb4:
  assert(%8-%9 = 0);

== Crab analyzer options == 
Abstract domain: Zones
Run checker: 1

=== Verification results === 
1  Number of total safe checks
0  Number of total error checks
0  Number of total warning checks
0  Number of total unreachable checks

=== Invariants that hold at the entry of each block === 
^bb0: {}
^bb1: {%0 -> [0, 10], %1 -> [0, 10], %1-%0<=0, %0-%1<=0}
^bb3: {%0 -> [0, 10], %1 -> [0, 10], %4 -> [0, 10], %5 -> [0, 10], %1-%0<=0, %4-%0<=0, %5-%0<=0, %0-%1<=0, %4-%1<=0, %5-%1<=0, %0-%4<=0, %1-%4<=0, %5-%4<=0, %1-%5<=0, %0-%5<=0, %4-%5<=0}
^bb4: {%0 -> [10, 10], %1 -> [10, 10], %4 -> [10, 10], %5 -> [10, 10], %8 -> [10, 10], %9 -> [10, 10], %1-%0<=0, %4-%0<=0, %5-%0<=0, %0-%1<=0, %4-%1<=0, %5-%1<=0, %0-%4<=0, %1-%4<=0, %5-%4<=0, %1-%5<=0, %0-%5<=0, %4-%5<=0}
^bb2: {%0 -> [0, 10], %1 -> [0, 10], %2 -> [0, 10], %3 -> [0, 10], %1-%0<=0, %2-%0<=0, %3-%0<=0, %0-%1<=0, %2-%1<=0, %3-%1<=0, %0-%2<=0, %1-%2<=0, %3-%2<=0, %1-%3<=0, %0-%3<=0, %2-%3<=0}
```


## Contributors
* Arie Gurfinkel <arie.gurfinkel@uwaterloo.ca> 
* Joseph Tafese <jetafese@uwaterloo.ca> 
* Jorge Navas <navasjorgea@gmail.com>
