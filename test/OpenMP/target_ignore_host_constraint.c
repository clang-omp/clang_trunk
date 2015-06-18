// RUN:   %clang -fopenmp -target x86_64-pc-linux-gnu -omptargets=nvptx64sm_35-nvidia-cuda \
// RUN:   -O3 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -input-file=target_ignore_host_constraint.ll.tgt-nvptx64sm_35-nvidia-cuda %s

int foo (float myvar)
{
  int a;
  __asm ("pmovmskb %1, %0" : "=r" (a) : "x" (myvar));
  return a & 0x8;
}

int bar (int a, float b){
  int c = a + foo(b);

  // Make sure we generate a target region instead of just crashing because ASM
  // constraints are not understood by the target
  // CHECK: define void @__omptgt__0_{{[0-9a-f]+_[0-9a-f]+}}_(
  // CHECK-DAG: float*
  // CHECK-DAG: i32*
  // CHECK: {
  // CHECK-NEXT: {{[a-zA-Z0-9_\.]+}}:
  #pragma omp target
    b = a+1;

  return b;
}
