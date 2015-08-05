// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() { }

int main(int argc, char **argv) {
    foo();
  #pragma omp target exit data // expected-error {{expected at least one map clause in OpenMP 'target exit data' construct}}

  return 0;
}
