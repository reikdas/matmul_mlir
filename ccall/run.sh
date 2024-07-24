#!/bin/bash
export LLVM_BUILD_DIR=/home/reikdas/llvm-project/build
${LLVM_BUILD_DIR}/bin/mlir-opt --convert-scf-to-cf --llvm-request-c-wrappers --convert-func-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts claude_mm.mlir | ${LLVM_BUILD_DIR}/bin/mlir-translate --mlir-to-llvmir > claude_mm.ll
${LLVM_BUILD_DIR}/bin/llc -filetype=obj claude_mm.ll -o claude_mm.o
${LLVM_BUILD_DIR}/bin/clang++ claude_mm.o mlir_main2.cpp -o program
./program