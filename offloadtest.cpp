#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <complex>
#include <dlfcn.h>
#include <sys/time.h>

typedef void gemm_t(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__);
typedef void symm_t(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__);
typedef void syrk_t(void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__, void *__restrict__);

int
main()
{
  void *cublas = dlopen("./liboffloadcublas.so", RTLD_NOW);
  if (cublas == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  /*
  void *iomp5 = dlopen("/opt/intel/composerxe/lib/intel64/libiomp5.so",
                       RTLD_LAZY | RTLD_GLOBAL);
  if (iomp5 == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  void *mkl_core = dlopen("/opt/intel/mkl/lib/intel64/libmkl_core.so",
                          RTLD_LAZY | RTLD_GLOBAL);
  if (mkl_core == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  void *mkl_intel_thread = dlopen("/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so",
                                  RTLD_LAZY | RTLD_GLOBAL);
  if (mkl_intel_thread == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  void *mkl_gf_lp64 = dlopen("/opt/intel/mkl/lib/intel64/libmkl_gf_lp64.so",
                             RTLD_LAZY | RTLD_GLOBAL);
  if (mkl_gf_lp64 == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  */
  void *libf77blas = dlopen("libf77blas.so.3",
                          RTLD_LAZY | RTLD_GLOBAL);
  if (libf77blas == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }

  gemm_t *cublas_sgemm = (gemm_t *)dlsym(cublas, "sgemm_");
  if (cublas_sgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *cublas_dgemm = (gemm_t *)dlsym(cublas, "dgemm_");
  if (cublas_dgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *cublas_cgemm = (gemm_t *)dlsym(cublas, "cgemm_");
  if (cublas_cgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *cublas_zgemm = (gemm_t *)dlsym(cublas, "zgemm_");
  if (cublas_zgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *cublas_ssymm = (symm_t *)dlsym(cublas, "ssymm_");
  if (cublas_ssymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *cublas_dsymm = (symm_t *)dlsym(cublas, "dsymm_");
  if (cublas_dsymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *cublas_csymm = (symm_t *)dlsym(cublas, "csymm_");
  if (cublas_csymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *cublas_zsymm = (symm_t *)dlsym(cublas, "zsymm_");
  if (cublas_zsymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *cublas_chemm = (symm_t *)dlsym(cublas, "chemm_");
  if (cublas_chemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *cublas_zhemm = (symm_t *)dlsym(cublas, "zhemm_");
  if (cublas_zhemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *cublas_ssyrk = (syrk_t *)dlsym(cublas, "ssyrk_");
  if (cublas_ssyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *cublas_dsyrk = (syrk_t *)dlsym(cublas, "dsyrk_");
  if (cublas_dsyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *cublas_csyrk = (syrk_t *)dlsym(cublas, "csyrk_");
  if (cublas_csyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *cublas_zsyrk = (syrk_t *)dlsym(cublas, "zsyrk_");
  if (cublas_zsyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *mkl_sgemm = (gemm_t *)dlsym(libf77blas, "sgemm_");
  if (mkl_sgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *mkl_dgemm = (gemm_t *)dlsym(libf77blas, "dgemm_");
  if (mkl_dgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *mkl_cgemm = (gemm_t *)dlsym(libf77blas, "cgemm_");
  if (mkl_cgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  gemm_t *mkl_zgemm = (gemm_t *)dlsym(libf77blas, "zgemm_");
  if (mkl_zgemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *mkl_ssymm = (symm_t *)dlsym(libf77blas, "ssymm_");
  if (mkl_ssymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *mkl_dsymm = (symm_t *)dlsym(libf77blas, "dsymm_");
  if (mkl_dsymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *mkl_csymm = (symm_t *)dlsym(libf77blas, "csymm_");
  if (mkl_csymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *mkl_zsymm = (symm_t *)dlsym(libf77blas, "zsymm_");
  if (mkl_zsymm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *mkl_chemm = (symm_t *)dlsym(libf77blas, "chemm_");
  if (mkl_chemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  symm_t *mkl_zhemm = (symm_t *)dlsym(libf77blas, "zhemm_");
  if (mkl_zhemm == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *mkl_ssyrk = (syrk_t *)dlsym(libf77blas, "ssyrk_");
  if (mkl_ssyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *mkl_dsyrk = (syrk_t *)dlsym(libf77blas, "dsyrk_");
  if (mkl_dsyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *mkl_csyrk = (syrk_t *)dlsym(libf77blas, "csyrk_");
  if (mkl_csyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }
  syrk_t *mkl_zsyrk = (syrk_t *)dlsym(libf77blas, "zsyrk_");
  if (mkl_zsyrk == 0) {
    fprintf(stderr, "%s\n", dlerror());
  }

  while (true) {
    char type;
    const char *routine;
    gemm_t *cublas_gemm, *mkl_gemm;
    symm_t *cublas_symm, *mkl_symm;
    syrk_t *cublas_syrk, *mkl_syrk;
    void *alpha, *beta;
    float salpha, sbeta;
    double dalpha, dbeta;
    std::complex<float> calpha, cbeta;
    std::complex<double> zalpha, zbeta;
    size_t typesize;
    switch (rand() % 4) {
    case 0:
      type = 's';
      routine = "ssymm";
      cublas_gemm = cublas_sgemm;
      cublas_symm = cublas_ssymm;
      cublas_syrk = cublas_ssyrk;
      mkl_gemm = mkl_sgemm;
      mkl_symm = mkl_ssymm;
      mkl_syrk = mkl_ssyrk;
      salpha = float(rand() / double(RAND_MAX));
      sbeta = (rand () % 2 == 0) ? 0.0f : float(rand() / double(RAND_MAX));
      alpha = &salpha;
      beta = &sbeta;
      typesize = sizeof(float);
      break;
    case 1:
      type = 'd';
      routine = "dsymm";
      cublas_gemm = cublas_dgemm;
      cublas_symm = cublas_dsymm;
      cublas_syrk = cublas_dsyrk;
      mkl_gemm = mkl_dgemm;
      mkl_symm = mkl_dsymm;
      mkl_syrk = mkl_dsyrk;
      dalpha = rand() / double(RAND_MAX);
      dbeta = (rand () % 2 == 0) ? 0.0 : (rand() / double(RAND_MAX));
      alpha = &dalpha;
      beta = &dbeta;
      typesize = sizeof(double);
      break;
    case 2:
      type = 'c';
      routine = "csymm";
      cublas_gemm = cublas_cgemm;
      cublas_symm = cublas_csymm;
      cublas_syrk = cublas_csyrk;
      mkl_gemm = mkl_cgemm;
      mkl_symm = mkl_csymm;
      mkl_syrk = mkl_csyrk;
      calpha = std::complex<float>(rand() / double(RAND_MAX),
                                   rand() / double(RAND_MAX));
      cbeta = (rand () % 2 == 0) ? 0.0f :
        std::complex<float>(rand() / double(RAND_MAX),
                            rand() / double(RAND_MAX));
      alpha = &calpha;
      beta = &cbeta;
      typesize = sizeof(std::complex<float>);
      break;
    case 3:
      type = 'z';
      routine = "zsymm";
      cublas_gemm = cublas_zgemm;
      cublas_symm = cublas_zsymm;
      cublas_syrk = cublas_zsyrk;
      mkl_gemm = mkl_zgemm;
      mkl_symm = mkl_zsymm;
      mkl_syrk = mkl_zsyrk;
      zalpha = std::complex<double>(rand() / double(RAND_MAX),
                                    rand() / double(RAND_MAX));
      zbeta = (rand () % 2 == 0) ? 0.0f :
        std::complex<double>(rand() / double(RAND_MAX),
                             rand() / double(RAND_MAX));
      alpha = &zalpha;
      beta = &zbeta;
      typesize = sizeof(std::complex<double>);
      break;
    case 4:
      type = 'c';
      routine = "chemm";
      cublas_gemm = cublas_cgemm;
      cublas_symm = cublas_chemm;
      mkl_gemm = mkl_cgemm;
      mkl_symm = mkl_chemm;
      calpha = std::complex<float>(rand() / double(RAND_MAX),
                                   rand() / double(RAND_MAX));
      cbeta = (rand () % 2 == 0) ? 0.0f :
        std::complex<float>(rand() / double(RAND_MAX),
                            rand() / double(RAND_MAX));
      alpha = &calpha;
      beta = &cbeta;
      typesize = sizeof(std::complex<float>);
      break;
    case 5:
      type = 'z';
      routine = "zhemm";
      cublas_gemm = cublas_zgemm;
      cublas_symm = cublas_zhemm;
      mkl_gemm = mkl_zgemm;
      mkl_symm = mkl_zhemm;
      zalpha = std::complex<double>(rand() / double(RAND_MAX),
                                    rand() / double(RAND_MAX));
      zbeta = (rand () % 2 == 0) ? 0.0f :
        std::complex<double>(rand() / double(RAND_MAX),
                             rand() / double(RAND_MAX));
      alpha = &zalpha;
      beta = &zbeta;
      typesize = sizeof(std::complex<double>);
      break;
    default:
      exit(-1);
      return -1;
    }
    /*
    char transa;
    switch (rand() % 3) {
    case 0:
      transa = 'N';
      break;
    case 1:
      transa = 'C';
      break;
    case 2:
      transa = 'T';
      break;
    default:
      exit(-1);
      return -1;
    }
    */
    char transa;
    if (type == 's' || type == 'd') {
    switch (rand() % 3) {
    case 0:
      transa = 'N';
      break;
    case 1:
      transa = 'C';
      break;
    case 2:
      transa = 'T';
      break;
    default:
      exit(-1);
      return -1;
    }
    } else {
    switch (rand() % 2) {
    case 0:
      transa = 'N';
      break;
    case 1:
      transa = 'T';
      break;
    default:
      exit(-1);
      return -1;
    }
    }
    char transb;
    switch (rand() % 3) {
    case 0:
      transb = 'N';
      break;
    case 1:
      transb = 'C';
      break;
    case 2:
      transb = 'T';
      break;
    default:
      exit(-1);
      return -1;
    }
    char side;
    switch (rand() % 2) {
    case 0:
      side = 'L';
      break;
    case 1:
      side = 'R';
      break;
    default:
      exit(-1);
      return -1;
    }
    char uplo;
    switch (rand() % 2) {
    case 0:
      uplo = 'U';
      break;
    case 1:
      uplo = 'L';
      break;
    default:
      exit(-1);
      return -1;
    }
    size_t m, n, k;
    do {
      /*
      m = rand() % 4000000 + 1000;
      */
      n = rand() % 4000000 + 1000;
      k = rand() % 4000000 + 1000;
      /*
      k = (side == 'L') ? m : n;
    } while ((m * k + k * n + 2llu * m * n) * typesize >
             12llu * 1024llu * 1024llu * 1024llu);
    } while ((k * k + 3llu * m * n) * typesize >
             12llu * 1024llu * 1024llu * 1024llu);
      */
    } while ((n * k + 2llu * n * n) * typesize >
             1024llu * 1024llu * 1024llu);
    /*
    printf("%cgemm(%c, %c, %d, %d, %d)\n", type, transa, transb, m, n, k);
    void *a = malloc(m * k * typesize);
    void *b = malloc(k * n * typesize);
    void *cublas_c = malloc(m * n * typesize);
    void *mkl_c = malloc(m * n * typesize);
    printf("%s(%c, %c, %d, %d)\n", routine, side, uplo, m, n);
    void *a = malloc(k * k * typesize);
    void *b = malloc(m * n * typesize);
    void *cublas_c = malloc(m * n * typesize);
    void *mkl_c = malloc(m * n * typesize);
    */
    printf("%csyrk(%c, %c, %d, %d)\n", type, uplo, transa, n, k);
    void *a = malloc(n * k * typesize);
    void *cublas_c = malloc(n * n * typesize);
    void *mkl_c = malloc(n * n * typesize);
    /*
    for (ptrdiff_t i = 0; i < m * k; ++i) {
    for (ptrdiff_t i = 0; i < k * k; ++i) {
    */
    for (ptrdiff_t i = 0; i < n * k; ++i) {
      switch (type) {
      case 's':
        ((float *)a)[i] = float(rand() / double(RAND_MAX));
        break;
      case 'd':
        ((double *)a)[i] = rand() / double(RAND_MAX);
        break;
      case 'c':
        ((std::complex<float> *)a)[i] =
          std::complex<float>(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX));
        break;
      case 'z':
        ((std::complex<double> *)a)[i] =
          std::complex<double>(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX));
        break;
      default:
        exit(-1);
        return -1;
      }
    }
    /*
    for (ptrdiff_t i = 0; i < k * n; ++i) {
    */
    /*
    for (ptrdiff_t i = 0; i < m * n; ++i) {
      switch (type) {
      case 's':
        ((float *)b)[i] = float(rand() / double(RAND_MAX));
        break;
      case 'd':
        ((double *)b)[i] = rand() / double(RAND_MAX);
        break;
      case 'c':
        ((std::complex<float> *)b)[i] =
          std::complex<float>(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX));
        break;
      case 'z':
        ((std::complex<double> *)b)[i] =
          std::complex<double>(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX));
        break;
      default:
        exit(-1);
        return -1;
      }
    }
    for (ptrdiff_t i = 0; i < m * n; ++i) {
    */
    for (ptrdiff_t i = 0; i < n * n; ++i) {
      switch (type) {
      case 's':
        ((float *)cublas_c)[i] = ((float *)mkl_c)[i] =
          float(rand() / double(RAND_MAX));
        break;
      case 'd':
        ((double *)cublas_c)[i] = ((double *)mkl_c)[i] =
          rand() / double(RAND_MAX);
        break;
      case 'c':
        ((std::complex<float> *)cublas_c)[i] =
          ((std::complex<float> *)mkl_c)[i] =
          std::complex<float>(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX));
        break;
      case 'z':
        ((std::complex<double> *)cublas_c)[i] =
          ((std::complex<double> *)mkl_c)[i] =
          std::complex<double>(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX));
        break;
      default:
        exit(-1);
        return -1;
      }
    }
    /*
    int lda = (transa == 'N') ? m : k;
    int ldb = (transb == 'N') ? k : n;
    */
    /*
    int lda = k;
    int ldb = m;
    int ldc = m;
    */
    int lda = (transa == 'N') ? n : k;
    int ldc = n;
    struct timeval t0, t1, t2;
    gettimeofday(&t0, 0);
    /*
    cublas_gemm(&transa, &transb, &m, &n, &k,
                alpha, a, &lda, b, &ldb, beta, cublas_c, &ldc);
    */
    /*
    cublas_symm(&side, &uplo, &m, &n,
                alpha, a, &lda, b, &ldb, beta, cublas_c, &ldc);
    */
    cublas_syrk(&uplo, &transa, &n, &k,
                alpha, a, &lda, beta, cublas_c, &ldc);
    gettimeofday(&t1, 0);
    /*
    mkl_gemm(&transa, &transb, &m, &n, &k,
             alpha, a, &lda, b, &ldb, beta, mkl_c, &ldc);
    */
    /*
    mkl_symm(&side, &uplo, &m, &n,
             alpha, a, &lda, b, &ldb, beta, mkl_c, &ldc);
    */
    mkl_syrk(&uplo, &transa, &n, &k,
             alpha, a, &lda, beta, mkl_c, &ldc);
    gettimeofday(&t2, 0);
    double cublas_sec =
      t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec) * 1.0e-6;
    double mkl_sec =
      t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0e-6;
    double cublas_gflops =
      n/*2.0 * m*/ * n * k / (1024.0 * 1024.0 * 1024.0) / cublas_sec;
    double mkl_gflops =
      n/*2.0 * m*/ * n * k / (1024.0 * 1024.0 * 1024.0) / mkl_sec;
    double error = 0.0;
    /*
    for (ptrdiff_t i = 0; i < m * n; ++i) {
    */
    for (ptrdiff_t i = 0; i < n * n; ++i) {
      switch (type) {
      case 's':
        if (((float *)cublas_c)[i] + ((float *)mkl_c)[i] != 0.0f)  {
          error += std::abs(((float *)cublas_c)[i] - ((float *)mkl_c)[i]) *
            0.5 / std::abs(((float *)cublas_c)[i] + ((float *)mkl_c)[i]);
        }
        break;
      case 'd':
        if (((double *)cublas_c)[i] + ((double *)mkl_c)[i] != 0.0)  {
          error += std::abs(((double *)cublas_c)[i] - ((double *)mkl_c)[i]) *
            0.5 / std::abs(((double *)cublas_c)[i] + ((double *)mkl_c)[i]);
        }
        break;
      case 'c':
        if (((std::complex<float> *)cublas_c)[i] +
            ((std::complex<float> *)mkl_c)[i] != 0.0f) {
          error += std::abs(((std::complex<float> *)cublas_c)[i] -
                            ((std::complex<float> *)mkl_c)[i]) *
            0.5 / std::abs(((std::complex<float> *)cublas_c)[i] +
                           ((std::complex<float> *)mkl_c)[i]);
        }
        break;
      case 'z':
        if (((std::complex<double> *)cublas_c)[i] +
            ((std::complex<double> *)mkl_c)[i] != 0.0) {
          error += std::abs(((std::complex<double> *)cublas_c)[i] -
                            ((std::complex<double> *)mkl_c)[i]) *
            0.5 / std::abs(((std::complex<double> *)cublas_c)[i] +
                           ((std::complex<double> *)mkl_c)[i]);
        }
        break;
      default:
        exit(-1);
        return -1;
      }
    }
    printf("CUBLAS:\t%.6f\tsec\t(%.6f\tGFLOPS)\n", cublas_sec, cublas_gflops);
    printf("MKL:\t%.6f\tsec\t(%.6f\tGFLOPS)\n", mkl_sec, mkl_gflops);
    /*
    printf("Error:\t%e\n", error / (m * n));
    */
    printf("Error:\t%e\n", error / (n * n));
    free(mkl_c);
    free(cublas_c);
    /*
    free(b);
    */
    free(a);
  }
  return 0;
}
