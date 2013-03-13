/*
 * symm.c
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

#define XSYMM_SUB(X) X##symm_
#define XSYMM(X) XSYMM_SUB(X)
#define CUBLASXSYMM_SUB(XX) cublas##XX##symm
#define CUBLASXSYMM(XX) CUBLASXSYMM_SUB(XX)
#define CUBLASXGEMM_SUB(XX) cublas##XX##gemm
#define CUBLASXGEMM(XX) CUBLASXGEMM_SUB(XX)

void
XSYMM(X)(char *__restrict__ side_, char *__restrict__ uplo_, INTEGER *__restrict__ m_, INTEGER *__restrict__ n_, T *__restrict__ alpha_, T *__restrict__ a, INTEGER *__restrict__ lda_, T *__restrict__ b, INTEGER *__restrict__ ldb_, T *__restrict__ beta_, T *__restrict__ c, INTEGER *__restrict__ ldc_)
{
  INTEGER info;
  INTEGER m, n, nrowa;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasOperation_t transa, transb, *transagemm, *transbgemm;
  INTEGER lda, ldb, ldc;
  T alpha, beta;
  cublasHandle_t handle;
  size_t mb, nb, *nrowab;
  T *d_a, *d_b, *d_c, **d_agemm, **d_bgemm;
  size_t d_apitch, d_bpitch, d_cpitch, *d_apitchgemm, *d_bpitchgemm;
  ptrdiff_t im, in, ik, *irowa, *icola, *irowb, *icolb;
  size_t imb, inb, ikb, *irowab, *icolab, *irowbb, *icolbb;
  info = 0;
  m = *m_;
  n = *n_;
  transb = CUBLAS_OP_N;
  switch (*side_) {
  case 'l':
  case 'L':
    side = CUBLAS_SIDE_LEFT;
    transagemm = &transa;
    transbgemm = &transb;
    nrowa = m;
    nrowab = &mb;
    d_agemm = &d_a;
    d_apitchgemm = &d_apitch;
    d_bgemm = &d_b;
    d_bpitchgemm = &d_bpitch;
    irowa = &im;
    icola = &ik;
    irowb = &ik;
    icolb = &in;
    irowab = &imb;
    icolab = &ikb;
    irowbb = &ikb;
    icolbb = &inb;
    break;
  case 'r':
  case 'R':
    side = CUBLAS_SIDE_RIGHT;
    transagemm = &transb;
    transbgemm = &transa;
    nrowa = n;
    nrowab = &nb;
    d_agemm = &d_b;
    d_apitchgemm = &d_bpitch;
    d_bgemm = &d_a;
    d_bpitchgemm = &d_apitch;
    irowa = &ik;
    icola = &in;
    irowb = &im;
    icolb = &ik;
    irowab = &ikb;
    icolab = &inb;
    irowbb = &imb;
    icolbb = &ikb;
    break;
  default:
    info = 1;
    xerbla_(MACRO2STR(XX) "SYMM ", &info);
    return;
  }
  switch (*uplo_) {
  case 'u':
  case 'U':
    uplo = CUBLAS_FILL_MODE_UPPER;
    break;
  case 'l':
  case 'L':
    uplo = CUBLAS_FILL_MODE_LOWER;
    break;
  default:
    info = 2;
    xerbla_(MACRO2STR(XX) "SYMM ", &info);
    return;
  }
  lda = *lda_;
  ldb = *ldb_;
  ldc = *ldc_;
  if (m < 0) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (lda < 1 || lda < nrowa) {
    info = 7;
  } else if (ldb < 1 || ldb < m) {
    info = 9;
  } else if (ldc < 1 || ldc < m) {
    info = 12;
  }
  if (info != 0) {
    xerbla_(MACRO2STR(XX) "SYMM ", &info);
    return;
  }
  alpha = *alpha_;
  beta = *beta_;
  if (m == 0 || n == 0 || (EQUAL(alpha, 0.0) && EQUAL(beta, 1.0))) {
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
  if (m >= 2048 && n >= 2048) {
    handle = get_handle();
  } else {
    handle = 0;
  }
  if (handle != 0) {
    size_t nmb, nnb;
    INTEGER iter;
    nmb = 1;
    nnb = 1;
    for (iter = 0; iter < n + m; iter++) {
      cudaError_t error;
      d_a = 0;
      d_b = 0;
      d_c = 0;
      mb = (m + nmb - 1) / nmb;
      nb = (n + nnb - 1) / nnb;
      error = cudaMallocPitch((void **)&d_a, &d_apitch,
                               sizeof(T) * *nrowab, *nrowab);
      error = cudaMallocPitch((void **)&d_b, &d_bpitch,
                               sizeof(T) * mb, nb);
      error = cudaMallocPitch((void **)&d_c, &d_cpitch,
                               sizeof(T) * mb, nb);
      if (d_a == 0 || d_b == 0 || d_c == 0) {
        if (nb < mb) {
          nmb++;
        } else {
          nnb++;
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
        for (ik = 0; ik < nrowa; ik += *nrowab) {
          T *cura, *curb;
          ikb = (*nrowab <= nrowa - ik) ? *nrowab : (nrowa - ik);
          if ((uplo == CUBLAS_FILL_MODE_UPPER && *irowa <= *icola) ||
              (uplo == CUBLAS_FILL_MODE_LOWER && *irowa >= *icola)) {
            cura = a + *irowa + lda * *icola;
            transa = CUBLAS_OP_N;
            if (cura != lasta) {
              status = cublasSetMatrix(*irowab, *icolab, sizeof(T),
                                       cura, lda,
                                       d_a, d_apitch / sizeof(T));
              lasta = cura;
            }
          } else {
            cura = a + *icola + lda * *irowa;
            transa = CUBLAS_OP_T;
            if (cura != lasta) {
              status = cublasSetMatrix(*icolab, *irowab, sizeof(T),
                                       cura, lda,
                                       d_a, d_apitch / sizeof(T));
              lasta = cura;
            }
          }
          curb = b + *irowb + ldb * *icolb;
          if (curb != lastb) {
            status = cublasSetMatrix(*irowbb, *icolbb, sizeof(T),
                                     curb, ldb,
                                     d_b, d_bpitch / sizeof(T));
            lastb = curb;
          }
          if (*irowa == *icola) {
            CUBLASXSYMM(XX)(handle,
                            side, uplo,
                            imb, inb,
                            alpha_,
                            d_a, d_apitch / sizeof(T),
                            d_b, d_bpitch / sizeof(T),
                            &d_beta,
                            d_c, d_cpitch / sizeof(T));
          } else {
            CUBLASXGEMM(XX)(handle,
                            *transagemm, *transbgemm,
                            imb, inb, ikb,
                            alpha_,
                            *d_agemm, *d_apitchgemm / sizeof(T),
                            *d_bgemm, *d_bpitchgemm / sizeof(T),
                            &d_beta,
                            d_c, d_cpitch / sizeof(T));
          }
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
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0;
    if (f == 0) {
      f = get_symbol(MACRO2STR(X) "symm_");
    }
    (*f)(side_, uplo_, m_, n_, alpha_, a, lda_, b, ldb_, beta_, c, ldc_);
  }
}

#undef CUBLASXGEMM
#undef CUBLASXGEMM_SUB
#undef CUBLASXSYMM
#undef CUBLASXSYMM_SUB
#undef XSYMM
#undef XSYMM_SUB
