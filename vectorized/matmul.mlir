func.func @matmul(%M: i64, %N: i64, %K: i64, %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  
  %M_idx = arith.index_cast %M : i64 to index
  %N_idx = arith.index_cast %N : i64 to index
  %K_idx = arith.index_cast %K : i64 to index

  scf.for %i = %c0 to %M_idx step %c1 {
    scf.for %j = %c0 to %N_idx step %c1 {
      %init = arith.constant 0.0 : f32
      %sum_scalar = scf.for %k = %c0 to %K_idx step %c8 iter_args(%sum_iter = %init) -> (f32) {
        %A_idx = arith.muli %i, %K_idx : index
        %A_idx_k = arith.addi %A_idx, %k : index
        
        // Load vector from A
        %a_vec = vector.load %A[%A_idx_k] : memref<?xf32>, vector<8xf32>
        
        // Initialize b_vec
        %b_vec_init = vector.broadcast %init : f32 to vector<8xf32>
        
        // Load B column elements individually
        %b_vec = scf.for %l = %c0 to %c8 step %c1 iter_args(%b_vec_iter = %b_vec_init) -> (vector<8xf32>) {
          %k_plus_l = arith.addi %k, %l : index
          %B_idx = arith.muli %k_plus_l, %N_idx : index
          %B_idx_j = arith.addi %B_idx, %j : index
          %b_element = memref.load %B[%B_idx_j] : memref<?xf32>
          %b_vec_next = vector.insertelement %b_element, %b_vec_iter[%l : index] : vector<8xf32>
          scf.yield %b_vec_next : vector<8xf32>
        }
        
        // Compute dot product
        %dot_result = x86vector.avx.intr.dot %a_vec, %b_vec : vector<8xf32>
        
        %dot_sum = vector.reduction <add>, %dot_result : vector<8xf32> into f32
        
        // Accumulate the result
        %sum_next = arith.addf %sum_iter, %dot_sum : f32
        
        scf.yield %sum_next : f32
      }
      
      %C_idx = arith.muli %i, %N_idx : index
      %C_idx_j = arith.addi %C_idx, %j : index
      memref.store %sum_scalar, %C[%C_idx_j] : memref<?xf32>
    }
  }
  return
}