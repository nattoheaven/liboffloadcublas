/*
 * herk.c
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

#define XHERK_SUB(X) X##herk_
#define XHERK(X) XHERK_SUB(X)
#define CUBLASXHERK_SUB(XX) cublas##XX##herk
#define CUBLASXHERK(XX) CUBLASXHERK_SUB(XX)
#define CUBLASXGEMM_SUB(XX) cublas##XX##gemm
#define CUBLASXGEMM(XX) CUBLASXGEMM_SUB(XX)

void
XHERK(X)(char *__restrict__ uplo_, char *__restrict__ trans_, INTEGER *__restrict__ n_, INTEGER *__restrict__ k_, REALT *__restrict__ alpha_, T *__restrict__ a, INTEGER *__restrict__ lda_, REALT *__restrict__ beta_, T *__restrict__ c, INTEGER *__restrict__ ldc_)
{
  INTEGER info;
  INTEGER n, k, nrowa;
  cublasFillMode_t uplo;
  cublasOperation_t trans, transar;
  INTEGER lda, ldc;
  REALT alpha, beta;
  cublasHandle_t handle;
  size_t nb, kb, *nrowab, *ncolab;
  T *d_al, *d_ar, *d_c;
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
    xerbla_(MACRO2STR(XX) "HERK ", &info);
    return;
  }
  switch (*trans_) {
  case 'n':
  case 'N':
    trans = CUBLAS_OP_N;
    transar = CUBLAS_OP_C;
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
  default:
    info = 2;
    xerbla_(MACRO2STR(XX) "HERK ", &info);
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
    xerbla_(MACRO2STR(XX) "HERK ", &info);
    return;
  }
  alpha = *alpha_;
  beta = *beta_;
  if (n == 0 || ((REALEQUAL(alpha, 0.0) || k == 0) && REALEQUAL(beta, 1.0))) {
    return;
  }
  if (REALEQUAL(alpha, 0.0)) {
    ptrdiff_t i, j;
    if (REALEQUAL(beta, 0.0)) {
      for (j = 0; j < n; j++) {
        if (uplo == CUBLAS_FILL_MODE_UPPER) {
          memset(c + ldc * j, 0, sizeof(T) * (j + 1));
        } else {
          memset(c + ldc * j + j, 0, sizeof(T) * (n - j));
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        if (uplo == CUBLAS_FILL_MODE_UPPER) {
          for (i = 0; i < j; i++) {
            c[i + ldc * j] = REALMUL(beta, c[i + ldc * j]);
          }
          c[j + ldc * j] = LITERAL(beta * CREAL(c[j + ldc * j]));
        } else {
          c[j + ldc * j] = LITERAL(beta * CREAL(c[j + ldc * j]));
          for (i = j + 1; i < n; i++) {
            c[i + ldc * j] = REALMUL(beta, c[i + ldc * j]);
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
                               sizeof(T) * *nrowab, *ncolab);
      error = cudaMallocPitch((void **)&d_ar, &d_arpitch,
                               sizeof(T) * *nrowab, *ncolab);
      error = cudaMallocPitch((void **)&d_c, &d_cpitch,
                               sizeof(T) * nb, nb);
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
    T *lastal, *lastar;
    REALT d_beta;
    lastal = 0;
    lastar = 0;
    for (in = 0; in < n; in += nb) {
      inb = (nb <= n - in) ? nb : (n - in);
      for (jn = *jnmin; jn <= *jnmax; jn += nb) {
        cublasStatus_t status;
        jnb = (nb <= n - jn) ? nb : (n - jn);
        if (in == jn || !REALEQUAL(beta, 0.0)) {
          status = cublasSetMatrix(jnb, inb, sizeof(T),
                                   c + jn + ldc * in, ldc,
                                   d_c, d_cpitch / sizeof(T));
        }
        d_beta = beta;
        for (ik = 0; ik < k; ik += kb) {
          T *cural, *curar;
          ikb = (kb <= k - ik) ? kb : (k - ik);
          curar = a + *irowar + lda * *icolar;
          if (curar != lastar && (in != jn || curar != lastal)) {
            status = cublasSetMatrix(*irowarb, *icolarb, sizeof(T),
                                     curar, lda,
                                     d_ar,
                                     d_arpitch / sizeof(T));
            lastar = curar;
          }
          if (in == jn) {
            T *d_a;
            size_t d_apitch;
            if (curar == lastal) {
              d_a = d_al;
              d_apitch = d_alpitch;
            } else {
              d_a = d_ar;
              d_apitch = d_arpitch;
            }
            CUBLASXHERK(XX)(handle,
                            uplo, trans,
                            inb, ikb,
                            alpha_,
                            d_a, d_apitch / sizeof(T),
                            &d_beta,
                            d_c, d_cpitch / sizeof(T));
          } else {
            T d_alphac = LITERAL(*alpha_);
            T d_betac = LITERAL(d_beta);
            cural = a + *irowal + lda * *icolal;
            if (cural != lastal) {
              status = cublasSetMatrix(*irowalb, *icolalb, sizeof(T),
                                       cural, lda,
                                       d_al,
                                       d_alpitch / sizeof(T));
              lastal = cural;
            }
            CUBLASXGEMM(XX)(handle,
                            trans, transar,
                            jnb, inb, ikb,
                            &d_alphac,
                            d_al, d_alpitch / sizeof(T),
                            d_ar, d_arpitch / sizeof(T),
                            &d_betac,
                            d_c, d_cpitch / sizeof(T));
          }
          d_beta = (REALT) 1.0;
        }
        status = cublasGetMatrix(jnb, inb, sizeof(T),
                                 d_c, d_cpitch / sizeof(T),
                                 c + jn + ldc * in, ldc);
      }
    }
    error = cudaFree(d_c);
    error = cudaFree(d_ar);
    error = cudaFree(d_al);
  } else {
    static void (*f)(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__) = 0;
    if (f == 0) {
      f = get_symbol(MACRO2STR(X) "herk_");
    }
    (*f)(uplo_, trans_, n_, k_, alpha_, a, lda_, beta_, c, ldc_);
  }
}
