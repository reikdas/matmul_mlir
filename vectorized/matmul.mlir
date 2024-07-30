func.func @matmul(%M: i64, %N: i64, %K: i64, %A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  
  %M_idx = arith.index_cast %M : i64 to index
  %N_idx = arith.index_cast %N : i64 to index
  %K_idx = arith.index_cast %K : i64 to index

  %init = arith.constant 0.0 : f32
  %i0 = arith.constant 0 : i32
  %i4 = arith.constant 4 : i32

  scf.for %l = %c0 to %N_idx step %c8 {
    scf.for %i = %c0 to %M_idx step %c1 {
      %A1 = arith.muli %i, %N_idx : index
      %A2 = arith.addi %A1, %l : index

      // Load vector from A
      %a_vec = vector.load %A[%A2] : memref<?xf32>, vector<8xf32>

      // Initialize b_vec
      %b_vec_init = vector.broadcast %init : f32 to vector<8xf32>

      scf.for %j = %c0 to %K_idx step %c1 {
        %row_end = arith.addi %l, %c8 : index
        %b_vec = scf.for %k = %l to %row_end step %c1 iter_args(%b_vec_iter = %b_vec_init) -> (vector<8xf32>) {
          %B1 = arith.muli %k, %K_idx : index
          %B2 = arith.addi %B1, %j : index
          %b_element = memref.load %B[%B2] : memref<?xf32>
          %b1 = arith.subi %k, %l : index
          %b_vec_next = vector.insertelement %b_element, %b_vec_iter[%b1: index] : vector<8xf32>
          scf.yield %b_vec_next : vector<8xf32>
        }
        %dot_result = x86vector.avx.intr.dot %a_vec, %b_vec : vector<8xf32>
        // Have to extract upper and lower half of the dot result
        %1 = vector.extractelement %dot_result[%i0 : i32]: vector<8xf32>
        %2 = vector.extractelement %dot_result[%i4 : i32]: vector<8xf32>
        %dot_sum = arith.addf %1, %2 : f32
        %C1 = arith.muli %K_idx, %i : index
        %C2 = arith.addi %C1, %j : index
        %old_val = memref.load %C[%C2] : memref<?xf32>
        %new_val = arith.addf %old_val, %dot_sum : f32
        memref.store %new_val, %C[%C2] : memref<?xf32>
      }
    }
  }
  return
}
