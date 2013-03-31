/*
 * liboffloadcublas.c
 *
 * Copyright (C) 2013 NISHIMURA Ryōhei
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef INT64
#define INTEGER int64_t
#else
#define INTEGER int32_t
#endif

static cublasHandle_t handle = 0;

static void
*get_symbol(const char *symbol)
{
  void *f;
  ptrdiff_t i;
  static void **handles = 0;
  static size_t nhandles;

  if (handles == 0) {
    char *env, *filenames, *t, *p;
    size_t envlen;

    env = getenv("OFFLOADCUBLAS_SO");
    if (env == 0) {
      env = "/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_sequential.so:/opt/intel/mkl/lib/intel64/libmkl_gf_lp64.so";
    }
    envlen = strlen(env);
    filenames = malloc((envlen + 1) * sizeof(char));
    strcpy(filenames, env);
    handles = malloc(envlen * sizeof(void *));
    nhandles = 0;
    for (t = strtok_r(filenames, ":", &p); t != 0; t = strtok_r(0, ":", &p)) {
      handles[nhandles] = dlopen(t, RTLD_LAZY | RTLD_GLOBAL);
      if (handles[nhandles] == 0) {
        fprintf(stderr, "%s\n", dlerror());
      } else {
        nhandles++;
      }
    }
    free(filenames);
  }
  f = 0;
  for (i = nhandles - 1; i >= 0; --i) {
    f = dlsym(handles[i], symbol);
    if (f == 0) {
      fprintf(stderr, "%s\n", dlerror());
    } else {
      return f;
    }
  }
  exit(-1);
  return 0;
}

static cublasHandle_t
get_handle()
{
  if (handle == 0) {
    cudaError_t error;
    cublasStatus_t status;
    char *devicename;
    int ordinal;

    devicename = getenv("OFFLOADCUBLAS_DEVICE");
    if (devicename == 0) {
      ordinal = 0;
    } else {
      ordinal = atoi(devicename);
    }
    error = cudaSetDevice(ordinal);
    if (error != cudaSuccess) {
      fprintf(stderr, "cudaSetDevice: %s\n", cudaGetErrorString(error));
      return 0;
    }
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cublasCreate: %d\n", status);
      return 0;
    }

    status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cublasSetPointerMode: %d\n", status);
      return 0;
    }
  }
  return handle;
}

#define VOID2(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1)                    \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__) = 0;       \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1);                                                       \
  }

#define VOID4(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3);                                               \
  }

#define VOID5(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4);                                           \
  }

#define VOID6(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5);                                       \
  }

#define VOID7(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5, void *__restrict__ a6) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5, a6);                                   \
  }

#define VOID8(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5, void *__restrict__ a6, void *__restrict__ a7) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5, a6, a7);                               \
  }

#define VOID9(T, NAME)                                                  \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5, void *__restrict__ a6, void *__restrict__ a7, void *__restrict__ a8) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5, a6, a7, a8);                           \
  }

#define VOID10(T, NAME)                                                 \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5, void *__restrict__ a6, void *__restrict__ a7, void *__restrict__ a8, void *__restrict__ a9) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);                       \
  }

#define VOID11(T, NAME)                                                 \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5, void *__restrict__ a6, void *__restrict__ a7, void *__restrict__ a8, void *__restrict__ a9, void *__restrict__ a10) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);                  \
  }

#define VOID13(T, NAME)                                                 \
  void                                                                  \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4, void *__restrict__ a5, void *__restrict__ a6, void *__restrict__ a7, void *__restrict__ a8, void *__restrict__ a9, void *__restrict__ a10, void *__restrict__ a11, void *__restrict__ a12) \
  {                                                                     \
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    (*f)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);        \
  }

#define FUNC3(T, NAME)                                                  \
  T                                                                     \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2) \
  {                                                                     \
    static T (*f)(void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    return (*f)(a0, a1, a2);                                            \
  }

#define FUNC5(T, NAME)                                                  \
  T                                                                     \
  NAME(void *__restrict__ a0, void *__restrict__ a1, void *__restrict__ a2, void *__restrict__ a3, void *__restrict__ a4) \
  {                                                                     \
    static T (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0; \
    if (f == 0) {                                                       \
      f = get_symbol(#NAME);                                            \
    }                                                                   \
    return (*f)(a0, a1, a2, a3, a4);                                    \
  }

#define FULL(FUNC, NAME)                        \
  FUNC(float, s##NAME)                          \
  FUNC(double, d##NAME)                         \
  FUNC(cuFloatComplex, c##NAME)                 \
  FUNC(cuDoubleComplex, z##NAME)

#define REAL(FUNC, NAME)                        \
  FUNC(float, s##NAME)                          \
  FUNC(double, d##NAME)

#define CPLX(FUNC, NAME)                        \
  FUNC(cuFloatComplex, c##NAME)                 \
  FUNC(cuDoubleComplex, z##NAME)

VOID2(void, xerbla_);
REAL(VOID4, rotg_)
REAL(VOID5, rotmg_)
REAL(VOID7, rot_)
REAL(VOID6, rotm_)
FULL(VOID5, swap_)
FULL(VOID4, scal_)
VOID4(void, csscal_)
VOID4(void, zdscal_)
FULL(VOID5, copy_)
FULL(VOID6, axpy_)
REAL(FUNC5, dot_)
FUNC5(double, dsdot_)
CPLX(FUNC5, dotu_)
CPLX(FUNC5, dotc_)
FUNC5(float, sdsdot_)
REAL(FUNC3, nrm2_)
FUNC3(float, scnrm2_)
FUNC3(double, dznrm2_)
REAL(FUNC3, asum_)
FUNC3(float, scasum_)
FUNC3(double, dzasum_)
FUNC3(INTEGER, isamax_)
FUNC3(INTEGER, idamax_)
FUNC3(INTEGER, icamax_)
FUNC3(INTEGER, izamax_)
FULL(VOID11, gemv_)
FULL(VOID13, gbmv_)
CPLX(VOID10, hemv_)
CPLX(VOID11, hbmv_)
CPLX(VOID9, hpmv_)
REAL(VOID10, symv_)
REAL(VOID11, sbmv_)
REAL(VOID9, spmv_)
FULL(VOID8, trmv_)
FULL(VOID9, tbmv_)
FULL(VOID7, tpmv_)
FULL(VOID8, trsv_)
FULL(VOID9, tbsv_)
FULL(VOID7, tpsv_)
REAL(VOID9, ger_)
CPLX(VOID9, geru_)
CPLX(VOID9, gerc_)
CPLX(VOID7, her_)
CPLX(VOID6, hpr_)
CPLX(VOID9, her2_)
CPLX(VOID8, hpr2_)
REAL(VOID7, syr_)
REAL(VOID6, spr_)
REAL(VOID9, syr2_)
REAL(VOID8, spr2_)

#define MACRO2STR_SUB(X) #X
#define MACRO2STR(X) MACRO2STR_SUB(X)

#define X s
#define XX S
#define T float
#define TRANSC 1
#define EQUAL(A,B) ((A) == (float) (B))
#define MUL(A,B) ((A) * (B))
#define LITERAL(A) ((float) (A))
#include "gemm.c"
#include "symm.c"
#include "syrk.c"
#undef TRANSC
#undef LITERAL
#undef MUL
#undef EQUAL
#undef T
#undef XX
#undef X

#define X d
#define XX D
#define T double
#define TRANSC 1
#define EQUAL(A,B) ((A) == (double) (B))
#define MUL(A,B) ((A) * (B))
#define LITERAL(A) ((double) (A))
#include "gemm.c"
#include "symm.c"
#include "syrk.c"
#undef TRANSC
#undef LITERAL
#undef MUL
#undef EQUAL
#undef T
#undef XX
#undef X

#define X c
#define XX C
#define T cuFloatComplex
#define TRANSC 0
#define EQUAL(A,B) ((A).x == (float) (B) && (A).y == 0.0f)
#define MUL(A,B) (cuCmulf((A), (B)))
#define LITERAL(A) (make_cuFloatComplex((float) (A), 0.0f))
#define REALT float
#define CREAL(A) (cuCrealf(A))
#define REALEQUAL(A,B) ((A) == (float) (B))
#define REALMUL(A,B) (make_cuFloatComplex((float) (A) * cuCrealf(B), (float) (A) * cuCimagf(B)))
#include "gemm.c"
#include "symm.c"
#include "hemm.c"
#include "syrk.c"
#include "herk.c"
#undef REALMUL
#undef REALEQUAL
#undef CREAL
#undef REALT
#undef TRANSC
#undef LITERAL
#undef MUL
#undef EQUAL
#undef T
#undef XX
#undef X

#define X z
#define XX Z
#define T cuDoubleComplex
#define TRANSC 0
#define EQUAL(A,B) ((A).x == (double) (B) && (A).y == 0.0)
#define MUL(A,B) (cuCmul((A), (B)))
#define LITERAL(A) (make_cuDoubleComplex((double) (A), 0.0))
#define REALT double
#define CREAL(A) (cuCreal(A))
#define REALEQUAL(A,B) ((A) == (double) (B))
#define REALMUL(A,B) (make_cuDoubleComplex((double) (A) * cuCreal(B), (double) (A) * cuCimag(B)))
#include "gemm.c"
#include "symm.c"
#include "hemm.c"
#include "syrk.c"
#include "herk.c"
#undef REALMUL
#undef REALEQUAL
#undef CREAL
#undef REALT
#undef TRANSC
#undef LITERAL
#undef MUL
#undef EQUAL
#undef T
#undef XX
#undef X

/*
void
ssyrk_(char *__restrict__ uplo_, char *__restrict__ trans_, INTEGER *__restrict__ n_, INTEGER *__restrict__ k_, float *__restrict__ alpha_, float *__restrict__ a, INTEGER *__restrict__ lda_, float *__restrict__ beta_, float *__restrict__ c, INTEGER *__restrict__ ldc_)
{
  INTEGER info;
  INTEGER n, k, nrowa;
  cublasFillMode_t uplo;
  cublasOperation_t trans, transar;
  INTEGER lda, ldc;
  float alpha, beta;
  cublasHandle_t handle;
  size_t nb, kb, *nrowab, *ncolab;
  float *d_al, *d_ar, *d_c;
  size_t d_alpitch, d_arpitch, d_cpitch;
  ptrdiff_t in, jn, ik, *jnmin, *jnmax, zero, n1,
    *irowal, *icolal, *irowar, *icolar;
  size_t inb, jnb, ikb, *irowalb, *icolalb, *irowarb, *icolarb;
  info = 0;
  n = *n_;
  k = *k_;
  zero = 0;
  n1 = n - 1;
  switch (*uplo_) {
  case 'u':
  case 'U':
    uplo = CUBLAS_FILL_MODE_UPPER;
    jnmin = &zero;
    jnmax = &in;
    break;
  case 'l':
  case 'L':
    uplo = CUBLAS_FILL_MODE_LOWER;
    jnmin = &in;
    jnmax = &n1;
    break;
  default:
    info = 1;
    xerbla_("SSYRK ", &info);
    return;
  }
  switch (*trans_) {
  case 'n':
  case 'N':
    trans = CUBLAS_OP_N;
    transar = CUBLAS_OP_T;
    nrowa = n;
    nrowab = &nb;
    ncolab = &kb;
    irowal = &jn;
    icolal = &ik;
    irowar = &in;
    icolar = &ik;
    irowalb = &jnb;
    icolalb = &ikb;
    irowarb = &inb;
    icolarb = &ikb;
    break;
  case 'c':
  case 'C':
    trans = CUBLAS_OP_C;
    transar = CUBLAS_OP_N;
    nrowa = k;
    nrowab = &kb;
    ncolab = &nb;
    irowal = &ik;
    icolal = &jn;
    irowar = &ik;
    icolar = &in;
    irowalb = &ikb;
    icolalb = &jnb;
    irowarb = &ikb;
    icolarb = &inb;
    break;
  case 't':
  case 'T':
    trans = CUBLAS_OP_T;
    transar = CUBLAS_OP_N;
    nrowa = k;
    nrowab = &kb;
    ncolab = &nb;
    irowal = &ik;
    icolal = &jn;
    irowar = &ik;
    icolar = &in;
    irowalb = &ikb;
    icolalb = &jnb;
    irowarb = &ikb;
    icolarb = &inb;
    break;
  default:
    info = 2;
    xerbla_("SSYRK ", &info);
    return;
  }
  lda = *lda_;
  ldc = *ldc_;
  if (n < 0) {
    info = 3;
  } else if (k < 0) {
    info = 4;
  } else if (lda < 1 || lda < nrowa) {
    info = 7;
  } else if (ldc < 1 || ldc < n) {
    info = 10;
  }
  if (info != 0) {
    xerbla_("SSYRK ", &info);
    return;
  }
  alpha = *alpha_;
  beta = *beta_;
  if (n == 0 || ((alpha == 0.0f || k == 0) && beta == 1.0f)) {
    return;
  }
  if (alpha == 0.0f) {
    ptrdiff_t i, j;
    if (beta == 0.0f) {
      for (j = 0; j < n; j++) {
        if (uplo == CUBLAS_FILL_MODE_UPPER) {
          memset(c + ldc * j, 0, sizeof(float) * (j + 1));
        } else {
          memset(c + ldc * j + j, 0, sizeof(float) * (n - j));
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        if (uplo == CUBLAS_FILL_MODE_UPPER) {
          for (i = 0; i <= j; i++) {
            c[i + ldc * j] *= beta;
          }
        } else {
          for (i = j; i < n; i++) {
            c[i + ldc * j] *= beta;
          }
        }
      }
    }
    return;
  }
  if (n >= 2048 && k >= 2048) {
    handle = get_handle();
  } else {
    handle = 0;
  }
  if (handle != 0) {
    size_t nnb, nkb;
    INTEGER iter;
    nnb = 1;
    nkb = 1;
    for (iter = 0; iter < n + k; iter++) {
      cudaError_t error;
      d_al = 0;
      d_ar = 0;
      d_c = 0;
      nb = (n + nnb - 1) / nnb;
      kb = (k + nkb - 1) / nkb;
      error = cudaMallocPitch((void **)&d_al, &d_alpitch,
                               sizeof(float) * *nrowab, *ncolab);
      error = cudaMallocPitch((void **)&d_ar, &d_arpitch,
                               sizeof(float) * *nrowab, *ncolab);
      error = cudaMallocPitch((void **)&d_c, &d_cpitch,
                               sizeof(float) * nb, nb);
      if (d_al == 0 || d_ar == 0 || d_c == 0) {
        if (nb < kb) {
          nkb++;
        } else {
          nnb++;
        }
        if (d_c != 0) {
          error = cudaFree(d_c);
          d_c = 0;
        }
        if (d_ar != 0) {
          error = cudaFree(d_ar);
          d_ar = 0;
        }
        if (d_al != 0) {
          error = cudaFree(d_al);
          d_al = 0;
        }
      } else {
        break;
      }
    }
  }
  if (handle != 0 && d_al != 0 && d_ar != 0 && d_c != 0) {
    cudaError_t error;
    float *lastal, *lastar;
    float d_beta;
    lastal = 0;
    lastar = 0;
    for (in = 0; in < n; in += nb) {
      inb = (nb <= n - in) ? nb : (n - in);
      for (jn = *jnmin; jn <= *jnmax; jn += nb) {
        cublasStatus_t status;
        jnb = (nb <= n - jn) ? nb : (n - jn);
        if (in == jn || beta != 0.0f) {
          status = cublasSetMatrix(jnb, inb, sizeof(float),
                                   c + jn + ldc * in, ldc,
                                   d_c, d_cpitch / sizeof(float));
        }
        d_beta = beta;
        for (ik = 0; ik < k; ik += kb) {
          float *cural, *curar;
          ikb = (kb <= k - ik) ? kb : (k - ik);
          curar = a + *irowar + lda * *icolar;
          if (curar != lastar && (in != jn || curar != lastal)) {
            status = cublasSetMatrix(*irowarb, *icolarb, sizeof(float),
                                     curar, lda,
                                     d_ar,
                                     d_arpitch / sizeof(float));
            lastar = curar;
          }
          if (in == jn) {
            float *d_a;
            size_t d_apitch;
            if (curar == lastal) {
              d_a = d_al;
              d_apitch = d_alpitch;
            } else {
              d_a = d_ar;
              d_apitch = d_arpitch;
            }
            cublasSsyrk(handle,
                        uplo, trans,
                        inb, ikb,
                        alpha_,
                        d_a, d_apitch / sizeof(float),
                        &d_beta,
                        (float *)d_c, d_cpitch / sizeof(float));
          } else {
            cural = a + *irowal + lda * *icolal;
            if (cural != lastal) {
              status = cublasSetMatrix(*irowalb, *icolalb, sizeof(float),
                                       cural, lda,
                                       d_al,
                                       d_alpitch / sizeof(float));
              lastal = cural;
            }
            cublasSgemm(handle,
                        trans, transar,
                        jnb, inb, ikb,
                        alpha_,
                        d_al, d_alpitch / sizeof(float),
                        d_ar, d_arpitch / sizeof(float),
                        &d_beta,
                        (float *)d_c, d_cpitch / sizeof(float));
          }
          d_beta = 1.0f;
        }
        status = cublasGetMatrix(jnb, inb, sizeof(float),
                                 d_c, d_cpitch / sizeof(float),
                                 c + jn + ldc * in, ldc);
      }
    }
    error = cudaFree(d_c);
    error = cudaFree(d_ar);
    error = cudaFree(d_al);
  } else {
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0;
    if (f == 0) {
      f = get_symbol("ssyrk_");
    }
    (*f)(uplo_, trans_, n_, k_, alpha_, a, lda_, beta_, c, ldc_);
  }
}
*/
