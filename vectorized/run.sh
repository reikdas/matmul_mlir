#!/bin/bash
export LLVM_BUILD_DIR=/home/reikdas/llvm-project/build
LD_LIBRARY_PATH=${LLVM_BUILD_DIR}/lib ${LLVM_BUILD_DIR}/bin/mlir-opt matmul.mlir -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm="enable-x86vector" -llvm-request-c-wrappers  -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts  | ${LLVM_BUILD_DIR}/bin/mlir-translate --mlir-to-llvmir > matmul.ll
${LLVM_BUILD_DIR}/bin/llc -mattr=+avx -filetype=obj matmul.ll -o matmul.o
${LLVM_BUILD_DIR}/bin/clang++ -O3 -march=native -mprefer-vector-width=512 -L${LLVM_BUILD_DIR}/lib -lmlir_c_runner_utils -lmlir_runner_utils matmul.o mlir_main2.cpp -o program
LD_LIBRARY_PATH=${LLVM_BUILD_DIR}/lib ./program $1 $2 $3
