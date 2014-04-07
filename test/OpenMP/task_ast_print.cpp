// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// expected-no-diagnostics

void foo() {}

template <class T, class S>
T tmain (T argc, S **argv) {
  T b = argc, c, d, e, f, g;
  T arr[25][123][2234];
  static T a;
// CHECK: static int a;
#pragma omp task
  a=2;
// CHECK-NEXT: #pragma omp task
// CHECK-NEXT: a = 2;
#pragma omp task if(b), final(a), untied, default(shared) mergeable, private(argc,b),firstprivate(argv, c),shared(d,f) depend(in:argc) depend(out : c) depend(inout:arr[1:argc][:][0:23], argc)
  foo();
// CHECK: #pragma omp task if(b) final(a) untied default(shared) mergeable private(argc,b) firstprivate(argv,c) shared(d,f) depend(in: argc) depend(out: c) depend(inout: arr[1:argc][0:123 - 0][0:23],argc)
// CHECK-NEXT: foo();
  return T();
}

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  float arr[25][123][2234];
  static int a;
// CHECK: static int a;
#pragma omp task
  a=2;
// CHECK-NEXT: #pragma omp task
// CHECK-NEXT: a = 2;
#pragma omp task if(b), final(a), untied, default(shared) mergeable, private(argc,b),firstprivate(argv, c),shared(d,f) depend(in:argc) depend(out : c) depend(inout:arr[1:argc][:][0:23], argc)
  foo();
// CHECK: #pragma omp task if(b) final(a) untied default(shared) mergeable private(argc,b) firstprivate(argv,c) shared(d,f) depend(in: argc) depend(out: c) depend(inout: arr[1:argc][0:123 - 0][0:23],argc)
// CHECK-NEXT: foo();
  return tmain(argc, argv);
}
