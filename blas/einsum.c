
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cblas.h>

/* ===== TYPE DEFINITIONS ===== */

typedef uint32_t IndexBitmap;

typedef struct {
    double *data;
    int ndim;
    int *shape;
    IndexBitmap indices;
} Matrix;

typedef enum {
    BLAS_DOT,      /* scalar dot product */
    BLAS_AXPY,     /* vector y := y + alpha*x */
    BLAS_GER,      /* outer product */
    BLAS_GEMV,     /* y := A*x or x := A^T*y */
    BLAS_GEMM,     /* C := A*B */
    BLAS_GENERAL   /* fallback: general tensor contraction */
} BLASPattern;

typedef struct {
    BLASPattern pattern;
    int has_output;
    int num_sum_indices;
    int num_free_indices;
} BLASAnalysis;


IndexBitmap literal_to_bitmap(const char *lit) {
    IndexBitmap bm = 0;
    if (!lit) return 0;
    for (const char *p = lit; *p; p++) {
        char c = *p;
        if (c >= 'a' && c <= 'z') {
            bm |= (1u << (c - 'a'));
        }
    }
    return bm;
}

void print_bitmap_indices(IndexBitmap bm) {
    int first = 1;
    putchar('{');
    for (int i = 0; i < 26; i++) {
        if (bm & (1u << i)) {
            if (!first) putchar(',');
            putchar('a' + i);
            first = 0;
        }
    }
    putchar('}');
}

void print_bitmap_binary(IndexBitmap bm) {
    putchar('0'); putchar('b');
    for (int i = 25; i >= 0; --i) {
        putchar((bm & (1u << i)) ? '1' : '0');
        if (i % 4 == 0 && i != 0) putchar('_');
    }
}

Matrix* matrix_create_nd(int ndim, const int *shape, const char *indices_str) {
    if (ndim <= 0 || !shape) return NULL;
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->ndim = ndim;
    m->shape = (int*)malloc(sizeof(int) * ndim);
    if (!m->shape) { free(m); return NULL; }
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        m->shape[i] = shape[i];
        if (shape[i] <= 0) { free(m->shape); free(m); return NULL; }
        total *= (size_t)m->shape[i];
    }
    m->indices = literal_to_bitmap(indices_str);
    m->data = (double*)calloc(total, sizeof(double));
    if (!m->data) { free(m->shape); free(m); return NULL; }
    return m;
}

void matrix_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    free(m->shape);
    free(m);
}

static size_t compute_offset(const Matrix *m, const int *idx) {
    size_t off = 0;
    for (int d = 0; d < m->ndim; d++) {
        off = off * (size_t)m->shape[d] + (size_t)idx[d];
    }
    return off;
}

void matrix_set_nd(Matrix *m, const int *idx, double v) {
    if (!m || !idx) return;
    for (int d = 0; d < m->ndim; d++) {
        if (idx[d] < 0 || idx[d] >= m->shape[d]) return;
    }
    size_t off = compute_offset(m, idx);
    m->data[off] = v;
}

double matrix_get_nd(const Matrix *m, const int *idx) {
    if (!m || !idx) return 0.0;
    for (int d = 0; d < m->ndim; d++) {
        if (idx[d] < 0 || idx[d] >= m->shape[d]) return 0.0;
    }
    size_t off = compute_offset(m, idx);
    return m->data[off];
}

double matrix_get(const Matrix *m, int i, int j) {
    if (!m || m->ndim != 2) return 0.0;
    int idx[2] = { i, j };
    return matrix_get_nd(m, idx);
}

void matrix_set(Matrix *m, int i, int j, double v) {
    if (!m || m->ndim != 2) return;
    int idx[2] = { i, j };
    matrix_set_nd(m, idx, v);
}

void next_indices(int *idx, const int *shape, int length, int *finished) {
    if (!idx || !shape || length <= 0) {
        if (finished) *finished = 1;
        return;
    }
    int j = length - 1;
    while (j >= 0 && (idx[j] + 1) >= shape[j]) {
        idx[j] = 0;
        j--;
    }
    if (j < 0) {
        if (finished) *finished = 1;
        return;
    }
    idx[j]++;
    if (finished) *finished = 0;
}

void matrix_print(const Matrix *m) {
    if (!m) { printf("(null)\n"); return; }
    if (m->ndim == 2) {
        printf("Matrix %dx%d indices=", m->shape[0], m->shape[1]);
    } else {
        printf("Matrix (ndim=%d) shape=({", m->ndim);
        for (int d = 0; d < m->ndim; d++) {
            if (d) printf(",");
            printf("%d", m->shape[d]);
        }
        printf("}) indices=");
    }
    print_bitmap_indices(m->indices);
    printf("\n");
    if (m->ndim == 2) {
        for (int r = 0; r < m->shape[0]; ++r) {
            for (int c = 0; c < m->shape[1]; ++c) {
                printf("%8.4f ", matrix_get(m, r, c));
            }
            printf("\n");
        }
    } else {
        size_t total = 1;
        for (int d = 0; d < m->ndim; d++) total *= (size_t)m->shape[d];
        for (size_t i = 0; i < total; i++) {
            printf("%8.4f ", m->data[i]);
            if ((i + 1) % 8 == 0) printf("\n");
        }
        printf("\n");
    }
}

void parse_einsum_notation(const char *notation, char *in1, char *in2, char *out) {
    const char *arrow = strstr(notation, "->");
    if (arrow) {
        strcpy(out, arrow + 2);
    } else {
        out[0] = '\0';
    }
    size_t left_len = arrow ? (size_t)(arrow - notation) : strlen(notation);
    char left[128];
    if (left_len >= sizeof(left)) left_len = sizeof(left) - 1;
    memcpy(left, notation, left_len);
    left[left_len] = '\0';
    char *comma = strchr(left, ',');
    if (comma) {
        *comma = '\0';
        strcpy(in1, left);
        strcpy(in2, comma + 1);
    } else {
        strcpy(in1, left);
        in2[0] = '\0';
    }
}

/* Permute tensor dimensions according to order array */
Matrix* matrix_permute(const Matrix *src, const int *order) {
    if (!src || !order) return NULL;
    
    int *new_shape = (int*)malloc(sizeof(int) * src->ndim);
    for (int i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[order[i]];
    }
    
    Matrix *dst = matrix_create_nd(src->ndim, new_shape, "per");
    free(new_shape);

    size_t total_elements = 1;
    for (int i = 0; i < src->ndim; i++) total_elements *= (size_t)src->shape[i];

    int *coords = (int*)calloc(src->ndim, sizeof(int));
    int *src_coords = (int*)malloc(sizeof(int) * src->ndim);

    for (size_t i = 0; i < total_elements; i++) {
        for (int d = 0; d < src->ndim; d++) {
            src_coords[order[d]] = coords[d];
        }

        double val = matrix_get_nd(src, src_coords);
        dst->data[i] = val;

        for (int d = src->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < dst->shape[d]) break;
            coords[d] = 0;
        }
    }

    free(coords);
    free(src_coords);
    return dst;
}


/* Count contracting indices between two index strings */
static int count_contraction_indices(const char *in1, const char *in2) {
    if (!in1 || !in2) return 0;
    int count = 0;
    for (const char *p = in1; *p; p++) {
        if (strchr(in2, *p)) count++;
    }
    return count;
}

/* Check if all indices in in1 appear in out */
static int all_indices_in(const char *in1, const char *out) {
    if (!in1 || !out) return 1;
    for (const char *p = in1; *p; p++) {
        if (!strchr(out, *p)) return 0;
    }
    return 1;
}

static BLASAnalysis analyze_blas_pattern(const char *notation, 
                                         const Matrix *A, 
                                         const Matrix *B) {
    BLASAnalysis result = {BLAS_GENERAL, 0, 0, 0};
    
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out);
    
    int ndim_A = A->ndim;
    int ndim_B = B->ndim;
    int ndim_out = (int)strlen(out);
    int in1_len = (int)strlen(in1);
    int in2_len = (int)strlen(in2);
    
    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    IndexBitmap bm_out = literal_to_bitmap(out);
    
    IndexBitmap input_mask = bm1 | bm2;
    IndexBitmap sum_mask = input_mask & (~bm_out);
    
    int num_sum_indices = 0;
    for (int i = 0; i < 26; i++) {
        if (sum_mask & (1u << i)) num_sum_indices++;
    }
    result.num_sum_indices = num_sum_indices;
    result.num_free_indices = ndim_out;
    result.has_output = (ndim_out > 0) ? 1 : 0;
    
    /* DOT: all indices match and contract to scalar (e.g., "i,i->", "ij,ij->") */
    if (ndim_out == 0 && ndim_A == ndim_B && in1_len == in2_len &&
        strcmp(in1, in2) == 0) {
        result.pattern = BLAS_DOT;
        return result;
    }
    
    /* AXPY: same indices, same output (element-wise add: "i,i->i", "ij,ij->ij") */
    if (strcmp(in1, in2) == 0 && strcmp(in1, out) == 0 && ndim_A == ndim_B) {
        result.pattern = BLAS_AXPY;
        return result;
    }
    
    /* GER: outer product - no shared indices, all appear in output (e.g., "i,j->ij", "ij,k->ijk") */
    if (count_contraction_indices(in1, in2) == 0 && 
        all_indices_in(in1, out) && all_indices_in(in2, out) &&
        ndim_out == in1_len + in2_len) {
        result.pattern = BLAS_GER;
        return result;
    }
    
    /* GEMV: exactly one index contracted, and B is a true vector (1D) */
    if (num_sum_indices == 1 && 
        ndim_out == (in1_len - 1 + in2_len - 1) &&
        B->ndim == 1) {
        result.pattern = BLAS_GEMV;
        return result;
    }
    
    /* GEMM: one or more contractions with free indices preserved (e.g., "ij,jc->ic", "ijk,kjl->ijl") */
    if (num_sum_indices >= 1 && 
        ndim_out == (in1_len + in2_len - 2 * num_sum_indices)) {
        result.pattern = BLAS_GEMM;
        return result;
    }
    
    result.pattern = BLAS_GENERAL;
    return result;
}


static double blas_dot(const double *x, int n, int incx,
                       const double *y, int incy) {
    return cblas_ddot(n, x, incx, y, incy);
}

static void blas_axpy(int n, double alpha, const double *x, int incx,
                      double *y, int incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}



static void blas_ger(int m, int n, double alpha, 
                     const double *x, int incx,
                     const double *y, int incy,
                     double *A, int ldA) {
    cblas_dger(CblasRowMajor, m, n, alpha, x, incx, y, incy, A, ldA);
}

static void blas_gemv(char trans, int m, int n, double alpha,
                      const double *A, int ldA,
                      const double *x, int incx, double beta,
                      double *y, int incy) {
    CBLAS_TRANSPOSE trans_flag = (trans == 'T') ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasRowMajor, trans_flag, m, n, alpha, A, ldA, x, incx, beta, y, incy);
}

static void blas_gemm(char transA, char transB, int m, int n, int k,
                      double alpha, const double *A, int ldA,
                      const double *B, int ldB, double beta,
                      double *C, int ldC) {
    CBLAS_TRANSPOSE trans_A = (transA == 'T') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = (transB == 'T') ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasRowMajor, trans_A, trans_B, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}



static Matrix* handle_dot(const Matrix *A, const Matrix *B) {
    /* DOT: contract all indices (e.g., "ij,ij->" or "i,i->") */
    if (A->ndim != B->ndim) {
        return NULL;
    }
    
    /* Check shapes match */
    for (int d = 0; d < A->ndim; d++) {
        if (A->shape[d] != B->shape[d]) return NULL;
    }
    
    /* Flatten to 1D vectors and use BLAS ddot */
    size_t total = 1;
    for (int d = 0; d < A->ndim; d++) total *= (size_t)A->shape[d];
    
    double dot_result = blas_dot(A->data, (int)total, 1, B->data, 1);
    
    int scalar_shape[1] = {1};
    Matrix *result = matrix_create_nd(1, scalar_shape, "");
    if (result) {
        result->data[0] = dot_result;
    }
    
    return result;
}

static Matrix* handle_axpy(const Matrix *A, const Matrix *B) {
    /* AXPY: element-wise add with same shape and indices (e.g., "i,i->i" or "ij,ij->ij") */
    if (A->ndim != B->ndim) {
        return NULL;
    }
    
    for (int d = 0; d < A->ndim; d++) {
        if (A->shape[d] != B->shape[d]) return NULL;
    }
    
    size_t total = 1;
    for (int d = 0; d < A->ndim; d++) total *= (size_t)A->shape[d];
    
    char indices_str[32] = "";
    for (int d = 0; d < A->ndim; d++) {
        indices_str[d] = 'a' + d;
    }
    indices_str[A->ndim] = '\0';
    
    Matrix *result = matrix_create_nd(A->ndim, A->shape, indices_str);
    if (!result) return NULL;
    
    memcpy(result->data, A->data, sizeof(double) * total);
    /* Use BLAS daxpy: y := y + alpha*x */
    blas_axpy((int)total, 1.0, B->data, 1, result->data, 1);
    
    return result;
}

static Matrix* handle_ger(const Matrix *A, const Matrix *B,
                          const char *in1, const char *in2, const char *out) {
    /* GER: outer product without shared indices using BLAS via permutation */
    
    /* For true outer product (both 1D), use BLAS directly */
    if (A->ndim == 1 && B->ndim == 1) {
        int m = A->shape[0];
        int n = B->shape[0];
        int out_shape[2] = {m, n};
        Matrix *result = matrix_create_nd(2, out_shape, out);
        if (!result) return NULL;
        
        blas_ger(m, n, 1.0, A->data, 1, B->data, 1, result->data, n);
        return result;
    }
    
    /* For higher-dimensional tensors, permute to 1D×1D, use BLAS, reshape back */
    size_t A_size = 1, B_size = 1;
    for (int i = 0; i < A->ndim; i++) A_size *= (size_t)A->shape[i];
    for (int i = 0; i < B->ndim; i++) B_size *= (size_t)B->shape[i];
    
    /* Create 1D versions by reshaping (permutation with single output dimension) */
    int shape_1d_A[1] = {(int)A_size};
    int shape_1d_B[1] = {(int)B_size};
    
    Matrix *A_1d = matrix_create_nd(1, shape_1d_A, "a");
    Matrix *B_1d = matrix_create_nd(1, shape_1d_B, "b");
    if (!A_1d || !B_1d) {
        matrix_free(A_1d);
        matrix_free(B_1d);
        return NULL;
    }
    
    /* Copy data (flatten) */
    memcpy(A_1d->data, A->data, A_size * sizeof(double));
    memcpy(B_1d->data, B->data, B_size * sizeof(double));
    
    /* Call BLAS outer product */
    int result_shape[2] = {(int)A_size, (int)B_size};
    Matrix *result_2d = matrix_create_nd(2, result_shape, "ab");
    if (!result_2d) {
        matrix_free(A_1d);
        matrix_free(B_1d);
        return NULL;
    }
    
    blas_ger((int)A_size, (int)B_size, 1.0, A_1d->data, 1, B_1d->data, 1, result_2d->data, (int)B_size);
    
    /* Reshape result back to n-dimensional */
    int out_shape[26];
    int out_len = (int)strlen(out);
    
    for (int i = 0; i < out_len; i++) {
        char idx_char = out[i];
        int found = 0;
        
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == idx_char) {
                out_shape[i] = A->shape[j];
                found = 1;
                break;
            }
        }
        
        if (!found) {
            for (int j = 0; j < B->ndim; j++) {
                if (in2[j] == idx_char) {
                    out_shape[i] = B->shape[j];
                    found = 1;
                    break;
                }
            }
        }
        
        if (!found) {
            matrix_free(A_1d);
            matrix_free(B_1d);
            matrix_free(result_2d);
            return NULL;
        }
    }
    
    Matrix *result = matrix_create_nd(out_len, out_shape, out);
    if (!result) {
        matrix_free(A_1d);
        matrix_free(B_1d);
        matrix_free(result_2d);
        return NULL;
    }
    
    /* Copy flattened 2D result to n-dimensional result */
    memcpy(result->data, result_2d->data, A_size * B_size * sizeof(double));
    
    matrix_free(A_1d);
    matrix_free(B_1d);
    matrix_free(result_2d);
    
    return result;
}

static Matrix* handle_gemv(const Matrix *A, const Matrix *B,
                           const char *in1, const char *in2, const char *out) {
    /* GEMV: contract one index where B is a vector (e.g., "ij,j->i", "ijk,k->ij") */
    
    /* B is guaranteed to be 1D at this point */
    if (B->ndim != 1) return NULL;
    
    /* Find the contraction index */
    char contract_idx = in2[0];  /* B has only one dimension, and it must be contracted */
    if (!strchr(in1, contract_idx)) return NULL;
    
    int contract_pos_A = -1;
    for (int i = 0; i < A->ndim; i++) {
        if (in1[i] == contract_idx) contract_pos_A = i;
    }
    if (contract_pos_A < 0) return NULL;
    if (A->shape[contract_pos_A] != B->shape[0]) return NULL;
    
    /* Build permutation for A: output indices first, then contracted */
    int perm_A[26];
    int p_a = 0;
    for (int i = 0; i < (int)strlen(out); i++) {
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == out[i]) {
                perm_A[p_a++] = j;
                break;
            }
        }
    }
    perm_A[p_a++] = contract_pos_A;
    
    /* Permute A */
    Matrix *A_perm = matrix_permute(A, perm_A);
    if (!A_perm) return NULL;
    
    /* Calculate BLAS gemv dimensions */
    int m = 1;
    for (int i = 0; i < (int)strlen(out); i++) {
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == out[i]) {
                m *= A->shape[j];
                break;
            }
        }
    }
    int k = B->shape[0];  /* contraction size */
    
    /* Build output shape */
    int out_shape[26];
    int out_len = (int)strlen(out);
    for (int i = 0; i < out_len; i++) {
        char idx_char = out[i];
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == idx_char) {
                out_shape[i] = A->shape[j];
                break;
            }
        }
    }
    
    Matrix *result = matrix_create_nd(out_len, out_shape, out);
    if (!result) {
        matrix_free(A_perm);
        return NULL;
    }
    
    /* Call BLAS GEMV: y := A*x where A is m×k, x is k, y is m */
    blas_gemv('N', m, k, 1.0, A_perm->data, k, B->data, 1, 0.0, result->data, 1);
    
    matrix_free(A_perm);
    return result;
}

static Matrix* handle_gemm(const Matrix *A, const Matrix *B, 
                           const char *in1, const char *in2, const char *out) {
    /* GEMM: contract one index with multiple free indices using BLAS via permutation */
    
    /* Find the contraction index */
    char contract_idx = 0;
    for (int i = 0; i < (int)strlen(in1); i++) {
        if (strchr(in2, in1[i]) && !strchr(out, in1[i])) {
            contract_idx = in1[i];
            break;
        }
    }
    if (!contract_idx) return NULL;
    
    /* Find positions of contraction index */
    int contract_pos_A = -1, contract_pos_B = -1;
    for (int i = 0; i < A->ndim; i++) {
        if (in1[i] == contract_idx) contract_pos_A = i;
    }
    for (int i = 0; i < B->ndim; i++) {
        if (in2[i] == contract_idx) contract_pos_B = i;
    }
    if (contract_pos_A < 0 || contract_pos_B < 0) return NULL;
    if (A->shape[contract_pos_A] != B->shape[contract_pos_B]) return NULL;
    
    int k = A->shape[contract_pos_A];  /* contraction size */
    
    /* Build permutation for A: free indices (from out) first, then contracted */
    int perm_A[26];
    int p_a = 0;
    for (int i = 0; i < (int)strlen(out); i++) {
        char out_idx = out[i];
        /* Find if this output index is in A */
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == out_idx) {
                perm_A[p_a++] = j;
                break;
            }
        }
    }
    perm_A[p_a++] = contract_pos_A;
    
    /* Build permutation for B: contracted index first, then free indices */
    int perm_B[26];
    int p_b = 0;
    perm_B[p_b++] = contract_pos_B;
    for (int i = 0; i < (int)strlen(out); i++) {
        char out_idx = out[i];
        /* Find if this output index is in B */
        for (int j = 0; j < B->ndim; j++) {
            if (in2[j] == out_idx && j != contract_pos_B) {
                perm_B[p_b++] = j;
                break;
            }
        }
    }
    
    /* Permute both inputs */
    Matrix *A_perm = matrix_permute(A, perm_A);
    Matrix *B_perm = matrix_permute(B, perm_B);
    if (!A_perm || !B_perm) {
        if (A_perm) matrix_free(A_perm);
        if (B_perm) matrix_free(B_perm);
        return NULL;
    }
    
    /* Calculate GEMM dimensions: C(m,n) = A(m,k) * B(k,n) */
    int m = 1;
    for (int i = 0; i < (int)strlen(out); i++) {
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == out[i]) {
                m *= A->shape[j];
                break;
            }
        }
    }
    
    int n = 1;
    for (int i = 0; i < (int)strlen(out); i++) {
        for (int j = 0; j < B->ndim; j++) {
            if (in2[j] == out[i]) {
                n *= B->shape[j];
                break;
            }
        }
    }
    
    /* Build output shape */
    int out_shape[26];
    int out_len = (int)strlen(out);
    for (int i = 0; i < out_len; i++) {
        char idx_char = out[i];
        int found = 0;
        
        for (int j = 0; j < A->ndim; j++) {
            if (in1[j] == idx_char) {
                out_shape[i] = A->shape[j];
                found = 1;
                break;
            }
        }
        
        if (!found) {
            for (int j = 0; j < B->ndim; j++) {
                if (in2[j] == idx_char) {
                    out_shape[i] = B->shape[j];
                    found = 1;
                    break;
                }
            }
        }
    }
    
    Matrix *result = matrix_create_nd(out_len, out_shape, out);
    if (!result) {
        matrix_free(A_perm);
        matrix_free(B_perm);
        return NULL;
    }
    
    /* Call BLAS: C(m,n) = A(m,k) * B(k,n) */
    blas_gemm('N', 'N', m, n, k, 1.0, A_perm->data, k, B_perm->data, n, 0.0, result->data, n);
    
    matrix_free(A_perm);
    matrix_free(B_perm);
    
    return result;
}


static Matrix* handle_general(const Matrix *A, const Matrix *B, 
                             const char *in1, const char *in2, const char *out) {


    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    IndexBitmap bmOut = literal_to_bitmap(out);

    IndexBitmap bmSum = (bm1 | bm2) & (~bmOut);

    char intermediate_indices[27];
    int *intermediate_shape = (int*)malloc(26 * sizeof(int));
    int idx_ptr = 0;

    int permA[10], permB[10]; 
    int countA_free = 0, countA_sum = 0;
    int countB_free = 0, countB_sum = 0;
    

    int pA_idx = 0;

    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (bmOut & (1u << (c-'a'))) permA[pA_idx++] = i;
    }
    countA_free = pA_idx;

    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (bmSum & (1u << (c-'a'))) permA[pA_idx++] = i;
    }
    countA_sum = pA_idx - countA_free;


    int pB_idx = 0;

    for (int k = countA_free; k < A->ndim; k++) {

        char target = in1[permA[k]];

        char *ptr = strchr(in2, target);
        if (ptr) permB[pB_idx++] = (int)(ptr - in2);
    }
    countB_sum = pB_idx;
    

    for (int i = 0; i < B->ndim; i++) {
        char c = in2[i];
        if (bmOut & (1u << (c-'a'))) permB[pB_idx++] = i;
    }
    countB_free = pB_idx - countB_sum;


    Matrix *permutedA = matrix_permute(A, permA);
    Matrix *permutedB = matrix_permute(B, permB);

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


    double *C_data = (double*)calloc(rows_A * cols_B, sizeof(double));


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
        
   
        if ((A_free & B_free) != 0) {
            free(C_data);
            return NULL;
        }
    }


    int intermediate_ndim = countA_free + countB_free;
    
    for (int i=0; i<countA_free; i++) {
        intermediate_indices[idx_ptr] = in1[permA[i]];
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

    /* Final permutation to match requested 'out' string */
    int final_perm[16];
    for (int i = 0; out[i]; i++) {
        char target = out[i];
        char *ptr = strchr(intermediate_indices, target);
        if (ptr) {
            final_perm[i] = (int)(ptr - intermediate_indices);
        } else {
            matrix_free(C_intermediate);
            return NULL;
        }
    }

    Matrix *FinalResult = matrix_permute(C_intermediate, final_perm);
    matrix_free(C_intermediate);
    return FinalResult;
}


Matrix* einsum(const char *notation, const Matrix *A, const Matrix *B) {
    if (!notation || !A || !B) return NULL;
    
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out);
    
    BLASAnalysis analysis = analyze_blas_pattern(notation, A, B);
    
    #ifdef PRINT_BLAS_ANALYSIS
    fprintf(stderr, "BLAS pattern analysis for '%s': ", notation);
    switch (analysis.pattern) {
        case BLAS_DOT:    fprintf(stderr, "DOT\n"); break;
        case BLAS_AXPY:   fprintf(stderr, "AXPY\n"); break;
        case BLAS_GER:    fprintf(stderr, "GER\n"); break;
        case BLAS_GEMV:   fprintf(stderr, "GEMV\n"); break;
        case BLAS_GEMM:   fprintf(stderr, "GEMM\n"); break;
        case BLAS_GENERAL:fprintf(stderr, "GENERAL (NOT SUPPORTED)\n"); break;
    }
    #endif
    
    switch (analysis.pattern) {
        case BLAS_DOT:
            return handle_dot(A, B);
        case BLAS_AXPY:
            return handle_axpy(A, B);
        case BLAS_GER:
            return handle_ger(A, B, in1, in2, out);
        case BLAS_GEMV:
            return handle_gemv(A, B, in1, in2, out);
        case BLAS_GEMM:
            return handle_gemm(A, B, in1, in2, out);
        case BLAS_GENERAL:
        default:
            /* Try general flattening approach for unsupported patterns */
            return handle_general(A, B, in1, in2, out);
    }
}
