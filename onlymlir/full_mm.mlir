module {

  func.func @matmul(%M: index, %N: index, %K: index, %A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      %zero = arith.constant 0.0 : f32
      scf.for %k = %c0 to %K step %c1 {
        %a = memref.load %A[%i, %k] : memref<?x?xf32>
        %b = memref.load %B[%k, %j] : memref<?x?xf32>
        %mul = arith.mulf %a, %b : f32
        %prev = memref.load %C[%i, %j] : memref<?x?xf32>
        %sum = arith.addf %prev, %mul : f32
        memref.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }

  return
}

  func.func @main() {
    %c2 = arith.constant 2 : index
    
    // Allocate memory for 2x2 matrices A, B, and C
    %A = memref.alloc(%c2, %c2) : memref<?x?xf32>
    %B = memref.alloc(%c2, %c2) : memref<?x?xf32>
    %C = memref.alloc(%c2, %c2) : memref<?x?xf32>

    // Initialize matrices A and B
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    %f4 = arith.constant 4.0 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Initialize matrix A
    memref.store %f1, %A[%c0, %c0] : memref<?x?xf32>
    memref.store %f2, %A[%c0, %c1] : memref<?x?xf32>
    memref.store %f3, %A[%c1, %c0] : memref<?x?xf32>
    memref.store %f4, %A[%c1, %c1] : memref<?x?xf32>

    // Initialize matrix B
    memref.store %f1, %B[%c0, %c0] : memref<?x?xf32>
    memref.store %f2, %B[%c0, %c1] : memref<?x?xf32>
    memref.store %f3, %B[%c1, %c0] : memref<?x?xf32>
    memref.store %f4, %B[%c1, %c1] : memref<?x?xf32>

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %zero = arith.constant 0.0 : f32
        memref.store %zero, %C[%i, %j] : memref<?x?xf32>
      }
    }

    // Call matmul function
    call @matmul(%c2, %c2, %c2, %A, %B, %C) : (index, index, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // Load C[0, 0]
    %result = memref.load %C[%c1, %c1] : memref<?x?xf32>
    
    // Allocate a small memref to hold the scalar
    %scalar_memref = memref.alloc() : memref<f32>

    // Store the result in the scalar memref
    memref.store %result, %scalar_memref[] : memref<f32>

    // Cast the scalar memref to memref<*xf32>
    %casted_result = memref.cast %scalar_memref : memref<f32> to memref<*xf32>

    // Print using printf
    call @printMemrefF32(%casted_result) : (memref<*xf32>) -> ()

    // Dealloc the scalar memref
    memref.dealloc %scalar_memref : memref<f32>
    // Free allocated memory
    memref.dealloc %A : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %C : memref<?x?xf32>

    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
}