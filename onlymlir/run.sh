#!/bin/bash
export LLVM_BUILD_DIR=/home/reikdas/llvm-project/build
${LLVM_BUILD_DIR}/bin/mlir-opt full_mm.mlir -pass-pipeline="builtin.module(func.func(convert-linalg-to-loops,convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" | ${LLVM_BUILD_DIR}/bin/mlir-translate --mlir-to-llvmir > full_mm.ll
${LLVM_BUILD_DIR}/bin/clang -L${LLVM_BUILD_DIR}/lib -lmlir_c_runner_utils -lmlir_runner_utils full_mm.ll -o program
LD_LIBRARY_PATH=${LLVM_BUILD_DIR}/lib ./program