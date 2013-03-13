/*
 * gemm.c
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

#define XGEMM_SUB(X) X##gemm_
#define XGEMM(X) XGEMM_SUB(X)
#define CUBLASXGEMM_SUB(XX) cublas##XX##gemm
#define CUBLASXGEMM(XX) CUBLASXGEMM_SUB(XX)

void
XGEMM(X)(char *__restrict__ transa_, char *__restrict__ transb_, INTEGER *__restrict__ m_, INTEGER *__restrict__ n_, INTEGER *__restrict__ k_, T *__restrict__ alpha_, T *__restrict__ a, INTEGER *__restrict__ lda_, T *__restrict__ b, INTEGER *__restrict__ ldb_, T *__restrict__ beta_, T *__restrict__ c, INTEGER *__restrict__ ldc_)
{
  INTEGER info;
  INTEGER m, n, k, nrowa, nrowb;
  cublasOperation_t transa, transb;
  INTEGER lda, ldb, ldc;
  T alpha, beta;
  cublasHandle_t handle;
  size_t mb, nb, kb, *nrowab, *ncolab, *nrowbb, *ncolbb;
  T *d_a, *d_b, *d_c;
  size_t d_apitch, d_bpitch, d_cpitch;
  ptrdiff_t im, in, ik, *irowa, *icola, *irowb, *icolb;
  size_t imb, inb, ikb, *irowab, *icolab, *irowbb, *icolbb;
  info = 0;
  m = *m_;
  n = *n_;
  k = *k_;
  switch (*transa_) {
  case 'n':
  case 'N':
    transa = CUBLAS_OP_N;
    nrowa = m;
    nrowab = &mb;
    ncolab = &kb;
    irowa = &im;
    icola = &ik;
    irowab = &imb;
    icolab = &ikb;
    break;
  case 'c':
  case 'C':
    transa = CUBLAS_OP_C;
    nrowa = k;
    nrowab = &kb;
    ncolab = &mb;
    irowa = &ik;
    icola = &im;
    irowab = &ikb;
    icolab = &imb;
    break;
  case 't':
  case 'T':
    transa = CUBLAS_OP_T;
    nrowa = k;
    nrowab = &kb;
    ncolab = &mb;
    irowa = &ik;
    icola = &im;
    irowab = &ikb;
    icolab = &imb;
    break;
  default:
    info = 1;
    xerbla_(MACRO2STR(XX) "GEMM ", &info);
    return;
  }
  switch (*transb_) {
  case 'n':
  case 'N':
    transb = CUBLAS_OP_N;
    nrowb = k;
    nrowbb = &kb;
    ncolbb = &nb;
    irowb = &ik;
    icolb = &in;
    irowbb = &ikb;
    icolbb = &inb;
    break;
  case 'c':
  case 'C':
    transb = CUBLAS_OP_C;
    nrowb = n;
    nrowbb = &nb;
    ncolbb = &kb;
    irowb = &in;
    icolb = &ik;
    irowbb = &inb;
    icolbb = &ikb;
    break;
  case 't':
  case 'T':
    transb = CUBLAS_OP_T;
    nrowb = n;
    nrowbb = &nb;
    ncolbb = &kb;
    irowb = &in;
    icolb = &ik;
    irowbb = &inb;
    icolbb = &ikb;
    break;
  default:
    info = 2;
    xerbla_(MACRO2STR(XX) "GEMM ", &info);
    return;
  }
  lda = *lda_;
  ldb = *ldb_;
  ldc = *ldc_;
  if (m < 0) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (k < 0) {
    info = 5;
  } else if (lda < 1 || lda < nrowa) {
    info = 8;
  } else if (ldb < 1 || ldb < nrowb) {
    info = 10;
  } else if (ldc < 1 || ldc < m) {
    info = 13;
  }
  if (info != 0) {
    xerbla_(MACRO2STR(XX) "GEMM ", &info);
    return;
  }
  alpha = *alpha_;
  beta = *beta_;
  if (m == 0 || n == 0 || ((EQUAL(alpha, 0.0) || k == 0) && EQUAL(beta, 1.0))) {
    return;
  }
  if (EQUAL(alpha, 0.0)) {
    ptrdiff_t i, j;
    if (EQUAL(beta, 0.0)) {
      for (j = 0; j < n; j++) {
        memset(c + ldc * j, 0, sizeof(T) * m);
      }
    } else {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          c[i + ldc * j] = MUL(beta, c[i + ldc * j]);
        }
      }
    }
    return;
  }
  if (m >= 2048 && n >= 2048 && k >= 2048) {
    handle = get_handle();
  } else {
    handle = 0;
  }
  if (handle != 0) {
    size_t nmb, nnb, nkb;
    INTEGER iter;
    nmb = 1;
    nnb = 1;
    nkb = 1;
    for (iter = 0; iter < n + m + k; iter++) {
      cudaError_t error;
      d_a = 0;
      d_b = 0;
      d_c = 0;
      mb = (m + nmb - 1) / nmb;
      nb = (n + nnb - 1) / nnb;
      kb = (k + nkb - 1) / nkb;
      error = cudaMallocPitch((void **)&d_a, &d_apitch,
                               sizeof(T) * *nrowab, *ncolab);
      error = cudaMallocPitch((void **)&d_b, &d_bpitch,
                               sizeof(T) * *nrowbb, *ncolbb);
      error = cudaMallocPitch((void **)&d_c, &d_cpitch,
                               sizeof(T) * mb, nb);
      if (d_a == 0 || d_b == 0 || d_c == 0) {
        if (nb < kb) {
          if (mb < kb) {
            nkb++;
          } else {
            nmb++;
          }
        } else {
          if (nb < mb) {
            nmb++;
          } else {
            nnb++;
          }
        }
        if (d_c != 0) {
          error = cudaFree(d_c);
          d_c = 0;
        }
        if (d_b != 0) {
          error = cudaFree(d_b);
          d_b = 0;
        }
        if (d_a != 0) {
          error = cudaFree(d_a);
          d_a = 0;
        }
      } else {
        break;
      }
    }
  }
  if (handle != 0 && d_a != 0 && d_b != 0 && d_c != 0) {
    cudaError_t error;
    T *lasta, *lastb;
    T d_beta;
    lasta = 0;
    lastb = 0;
    for (in = 0; in < n; in += nb) {
      inb = (nb <= n - in) ? nb : (n - in);
      for (im = 0; im < m; im += mb) {
        cublasStatus_t status;
        imb = (mb <= m - im) ? mb : (m - im);
        if (!EQUAL(beta, 0.0)) {
          status = cublasSetMatrix(imb, inb, sizeof(T),
                                   c + im + ldc * in, ldc,
                                   d_c, d_cpitch / sizeof(T));
        }
        d_beta = beta;
        for (ik = 0; ik < k; ik += kb) {
          T *cura, *curb;
          ikb = (kb <= k - ik) ? kb : (k - ik);
          cura = a + *irowa + lda * *icola;
          if (cura != lasta) {
            status = cublasSetMatrix(*irowab, *icolab, sizeof(T),
                                     cura, lda,
                                     d_a, d_apitch / sizeof(T));
            lasta = cura;
          }
          curb = b + *irowb + ldb * *icolb;
          if (curb != lastb) {
            status = cublasSetMatrix(*irowbb, *icolbb, sizeof(T),
                                     curb, ldb,
                                     d_b, d_bpitch / sizeof(T));
            lastb = curb;
          }
          CUBLASXGEMM(XX)(handle,
                          transa, transb,
                          imb, inb, ikb,
                          alpha_,
                          d_a, d_apitch / sizeof(T),
                          d_b, d_bpitch / sizeof(T),
                          &d_beta,
                          d_c, d_cpitch / sizeof(T));
          d_beta = LITERAL(1.0);
        }
        status = cublasGetMatrix(imb, inb, sizeof(T),
                                 d_c, d_cpitch / sizeof(T),
                                 c + im + ldc * in, ldc);
      }
    }
    error = cudaFree(d_c);
    error = cudaFree(d_b);
    error = cudaFree(d_a);
  } else {
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0;
    if (f == 0) {
      f = get_symbol(MACRO2STR(X) "gemm_");
    }
    (*f)(transa_, transb_, m_, n_, k_, alpha_, a, lda_, b, ldb_, beta_, c, ldc_);
  }
}

#undef CUBLASXGEMM
#undef CUBLASXGEMM_SUB
#undef XGEMM
#undef XGEMM_SUB
