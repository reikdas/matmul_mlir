func.func @matmul(%M: i64, %N: i64, %K: i64, %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index  // Vector width of 8 float32 elements (256 bits for AVX)
  %i0 = arith.constant 0 : i32
  %i4 = arith.constant 4 : i32
  
  // Convert i64 dimensions to index type
  
  %N_idx = arith.index_cast %N : i64 to index
  %K_idx = arith.index_cast %K : i64 to index

  // Nested loops for matrix multiplication
  scf.for %i = %c0 to %M_idx step %c1 {
    scf.for %j = %c0 to %N_idx step %c1 {
      %init = arith.constant 0.0 : f32
      %sum_scalar = scf.for %k = %c0 to %K_idx step %c8 iter_args(%sum_iter = %init) -> (f32) {
        // Compute base indices for A and B
        %A_idx = arith.muli %i, %K_idx : index
        %A_idx_k = arith.addi %A_idx, %k : index
        %B_idx = arith.muli %k, %N_idx : index
        %B_idx_j = arith.addi %B_idx, %j : index
        
        // Load vectors from A and B
        // %a_vec = vector.load %A[%A_idx_k] : memref<?xf32>, vector<8xf32>
        // %b_vec = vector.load %B[%B_idx_j] : memref<?xf32>, vector<8xf32>

        // vector.print %a_vec : vector<8xf32>
        // vector.print %b_vec : vector<8xf32>

        %a_vec = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : vector<8xf32>
        %b_vec = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
        
        // Compute dot product
        %dot_result = x86vector.avx.intr.dot %a_vec, %b_vec : vector<8xf32>

        // vector.print %dot_result : vector<8xf32>
        
        // // Extract and sum the relevant elements
        %sum1 = vector.extractelement %dot_result[%i0 : i32] : vector<8xf32>
        %sum2 = vector.extractelement %dot_result[%i4 : i32] : vector<8xf32>

        // vector.print %sum1 : vector<8xf32>
        // vector.print %sum2 : vector<8xf32>

        %dot_sum = arith.addf %sum1, %sum2 : f32
        
        // // Accumulate the result
        // %sum_next = arith.addf %sum_iter, %dot_sum : f32

    //     %f1 = arith.constant 1.0 : f32
    // %f2 = arith.constant 2.0 : f32
    //     %sum_next = arith.addf %f1, %f2 : f32
        
        scf.yield %sum_next : f32
      }
      
      // Compute index for C and store the result
      %C_idx = arith.muli %i, %N_idx : index
      %C_idx_j = arith.addi %C_idx, %j : index
      memref.store %sum_scalar, %C[%C_idx_j] : memref<?xf32>
    }
  }
  return
}