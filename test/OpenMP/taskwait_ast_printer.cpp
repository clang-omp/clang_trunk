// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <int a>
void here() {
#pragma omp taskwait
}

// CHECK: template <int a = 5> void here() {
// CHECK-NEXT:     #pragma omp taskwait
// CHECK-NEXT: }
// CHECK: template <int a> void here() {
// CHECK-NEXT:     #pragma omp taskwait
// CHECK-NEXT: }

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp taskwait
// CHECK-NEXT: #pragma omp taskwait

  here<5>();
// CHECK-NEXT: here<5>();

  return (0);
}

#endif
