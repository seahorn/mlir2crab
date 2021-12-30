# SeaHorn builder image that builds binary SeaHorn release package
# Primarily used by the CI
# Arguments:
#  - BASE-IMAGE: bionic-llvm10, focal-llvm10
#  - BUILD_TYPE: Debug, RelWithDebInfo, Coverage
ARG BASE_IMAGE=bionic-llvm10
##FROM agurfinkel/buildback-deps-btor2mlir
FROM seahorn/buildpack-deps-seahorn:$BASE_IMAGE

# Download mlir2crab
RUN cd / && rm -rf /opt/mlir2crab && cd /opt && \
    git clone https://github.com/seahorn/mlir2crab.git; \
    mkdir -p /opt/mlir2crab/debug    

# Download crab
RUN cd /opt && \
    git clone  https://github.com/seahorn/crab.git

# Build mlir2crab 
ARG BUILD_TYPE=Debug
WORKDIR /opt/mlir2crab/debug
RUN cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang-10 \
    -DCMAKE_CXX_COMPILER=clang++-10 \
    -DMLIR_DIR=/opt/llvm/run/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm/run/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DLLVM_ENABLE_LLD=ON \
    -DCRAB_DIR=/opt/crab \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run && \
    ninja install

ENV PATH "/usr/bin:$PATH"
ENV PATH "/opt/llvm/run/bin:$PATH"
ENV PATH "/opt/mlir2crab/debug/run/bin:$PATH"

# Run tests
RUN cmake --build . --target check-mlir2crab
