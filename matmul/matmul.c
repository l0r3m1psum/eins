#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include "../einsum.h"

// matrix_permute is defined in blas/einsum.c
extern Matrix* matrix_permute(const Matrix *src, const int *order);



Matrix* einsum_matmul(const char *notation, const Matrix *A, const Matrix *B) {
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out); //

    // --- ANALYZE INDICES ---
    // Identify Free indices (Keep) vs Summation indices (Contract)
    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    IndexBitmap bmOut = literal_to_bitmap(out);
    
    // Summation indices are those in (Input1 U Input2) - Output
    IndexBitmap bmSum = (bm1 | bm2) & (~bmOut);

    // Lists to store the permutation orders
    int permA[10], permB[10]; 
    int countA_free = 0, countA_sum = 0;
    int countB_free = 0, countB_sum = 0;
    
    // Build Permutation for A: [Free indices..., Summation indices...]
    // This groups all free dims to the left, and sum dims to the right.
    int pA_idx = 0;
    // Add A's free indices
    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (bmOut & (1u << (c-'a'))) permA[pA_idx++] = i;
    }
    countA_free = pA_idx;
    // Add A's summation indices
    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (bmSum & (1u << (c-'a'))) permA[pA_idx++] = i;
    }
    countA_sum = pA_idx - countA_free;

    // Build Permutation for B: [Summation indices..., Free indices...]
    // Note order: Sum indices on Left (to align with A's Right columns)
    int pB_idx = 0;
    // Add B's summation indices
    // CRITICAL: Must be in same relative order as they appear in A's permuted end
    // To match A's sum group, we must look at how we ordered A's sum group.
    // However, simplest is to just sort alphabetically or order by appearance. 
    // Let's iterate A's sum indices and find where they are in B.
    for (int k = countA_free; k < A->ndim; k++) {
        // The char at A's k-th permuted dimension
        char target = in1[permA[k]];
        // Find this char in B
        char *ptr = strchr(in2, target);
        if (ptr) permB[pB_idx++] = (int)(ptr - in2);
    }
    countB_sum = pB_idx;
    
    // Add B's free indices
    for (int i = 0; i < B->ndim; i++) {
        char c = in2[i];
        if (bmOut & (1u << (c-'a'))) permB[pB_idx++] = i;
    }
    countB_free = pB_idx - countB_sum;

    // --- PERMUTE TENSORS ---
    Matrix *permutedA = matrix_permute(A, permA);
    Matrix *permutedB = matrix_permute(B, permB);

    // --- CALCULATE FLATTENED DIMENSIONS ---
    int rows_A = 1; 
    for(int i=0; i<countA_free; i++) rows_A *= permutedA->shape[i];
    
    int cols_A_sum = 1;
    for(int i=countA_free; i<permutedA->ndim; i++) cols_A_sum *= permutedA->shape[i];
    
    int rows_B_sum = 1; // Should equal cols_A_sum
    for(int i=0; i<countB_sum; i++) rows_B_sum *= permutedB->shape[i];
    
    int cols_B = 1;
    for(int i=countB_sum; i<permutedB->ndim; i++) cols_B *= permutedB->shape[i];

    if (cols_A_sum != rows_B_sum) {
        fprintf(stderr, "Dimension mismatch in contraction: %d vs %d\n", cols_A_sum, rows_B_sum);
        return NULL;
    }

    // --- MATRIX MULTIPLICATION (rows_A x cols_A_sum) * (rows_B_sum x cols_B) ---
    // C_flat will be (rows_A x cols_B)
    double *C_data = (double*)calloc(rows_A * cols_B, sizeof(double));

    // Use BLAS for efficient matrix multiplication
    cblas_dgemm(CblasRowMajor,           // Row-major storage
                CblasNoTrans,             // No transpose on A
                CblasNoTrans,             // No transpose on B
                rows_A,                   // Number of rows in A and C
                cols_B,                   // Number of columns in B and C
                cols_A_sum,               // Number of columns in A / rows in B
                1.0,                      // alpha scaling factor
                permutedA->data,          // Matrix A
                cols_A_sum,               // Leading dimension of A
                permutedB->data,          // Matrix B
                cols_B,                   // Leading dimension of B
                0.0,                      // beta scaling factor (0 since C is zero-initialized)
                C_data,                   // Result matrix C
                cols_B);                  // Leading dimension of C

    // Cleanup permuted temporaries
    matrix_free(permutedA);
    matrix_free(permutedB);


    {
        IndexBitmap A_free = 0, B_free = 0;
        for (int i = 0; i < countA_free; i++) {
            char c = in1[permA[i]];
            A_free |= (1u << (c - 'a'));
        }
        for (int i = 0; i < countB_free; i++) {
            char c = in2[permB[countB_sum + i]];
            B_free |= (1u << (c - 'a'));
        }
        
        // If A_free and B_free overlap, reject (pattern like ijk,ikl->ijl)
        if ((A_free & B_free) != 0) {
            free(C_data);
            return NULL;
        }
    }
    
    // Construct the shape of the intermediate result
    int intermediate_ndim = countA_free + countB_free;
    int *intermediate_shape = (int*)malloc(sizeof(int) * intermediate_ndim);
    char intermediate_indices[32]; 
    int idx_ptr = 0;
    
    // Reconstruct the indices string for the intermediate result
    for (int i=0; i<countA_free; i++) {
        char original_char = in1[permA[i]];
        intermediate_indices[idx_ptr] = original_char;
        intermediate_shape[idx_ptr] = A->shape[permA[i]];
        idx_ptr++;
    }
    for (int i=0; i<countB_free; i++) {
        int original_dim_idx = permB[countB_sum + i]; 
        intermediate_indices[idx_ptr] = in2[original_dim_idx];
        intermediate_shape[idx_ptr] = B->shape[original_dim_idx];
        idx_ptr++;
    }
    intermediate_indices[idx_ptr] = '\0';

    Matrix *C_intermediate = NULL;
    if (intermediate_ndim <= 0) {
        // Scalar intermediate: represent as a 1-element 1-D matrix
        int one = 1;
        C_intermediate = matrix_create_nd(1, &one, "s");
        if (!C_intermediate) {
            free(intermediate_shape);
            free(C_data);
            return NULL;
        }
        free(C_intermediate->data);
        C_intermediate->data = C_data; 
        free(intermediate_shape);

        // If output is also scalar (out length 0), return directly
        if (out[0] == '\0') {
            return C_intermediate;
        
    } else {
        C_intermediate = matrix_create_nd(intermediate_ndim, intermediate_shape, intermediate_indices);
        if (!C_intermediate) {
            free(intermediate_shape);
            free(C_data);
            return NULL;
        }
        free(C_intermediate->data); 
        C_intermediate->data = C_data; 
        free(intermediate_shape);
    }

   
    int final_perm[16];
    for (int i = 0; out[i]; i++) {
        char target = out[i];
        char *ptr = strchr(intermediate_indices, target);
        if (ptr) {
            final_perm[i] = (int)(ptr - intermediate_indices);
        } else {
            // if target not found, default to 0
            final_perm[i] = 0;
        }
    }

    if (intermediate_ndim <= 0) {
        Matrix *FinalResult = matrix_permute(C_intermediate, final_perm);
        matrix_free(C_intermediate);
        return FinalResult;
    }

    Matrix *FinalResult = matrix_permute(C_intermediate, final_perm);
    matrix_free(C_intermediate);
    return FinalResult;
}