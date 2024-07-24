module {
  func.func @matmul(%M: i64, %N: i64, %K: i64, %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // Convert i64 dimensions to index type
    %M_idx = arith.index_cast %M : i64 to index
    %N_idx = arith.index_cast %N : i64 to index
    %K_idx = arith.index_cast %K : i64 to index

    // Nested loops for matrix multiplication
    scf.for %i = %c0 to %M_idx step %c1 {
      scf.for %j = %c0 to %N_idx step %c1 {
        %init = arith.constant 0.0 : f32
        %sum = scf.for %k = %c0 to %K_idx step %c1 iter_args(%sum_iter = %init) -> (f32) {
          // Compute indices for A and B
          %A_idx = arith.muli %i, %K_idx : index
          %A_idx_k = arith.addi %A_idx, %k : index
          %B_idx = arith.muli %k, %N_idx : index
          %B_idx_j = arith.addi %B_idx, %j : index
          
          // Load elements from A and B
          %a_elem = memref.load %A[%A_idx_k] : memref<?xf32>
          %b_elem = memref.load %B[%B_idx_j] : memref<?xf32>
          
          // Multiply and accumulate
          %mul = arith.mulf %a_elem, %b_elem : f32
          %sum_next = arith.addf %sum_iter, %mul : f32
          
          scf.yield %sum_next : f32
        }
        
        // Compute index for C and store the result
        %C_idx = arith.muli %i, %N_idx : index
        %C_idx_j = arith.addi %C_idx, %j : index
        memref.store %sum, %C[%C_idx_j] : memref<?xf32>
      }
    }
    return
  }
}