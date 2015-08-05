///
/// Perform several control loop code generation tests
/// (code within target region for nvptx)
///

///##############################################
///
/// Empty target region (control loop skeleton)
///
///##############################################

#ifdef TT1
// RUN:   %clang -fopenmp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-linux \
// RUN:   -DTT1 -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK1 -input-file=target_control_loop_codegen_for_c.ll.tgt-nvptx64sm_35-nvidia-linux %s

// CK1: @__omptgt__ControlState = common addrspace(3) global [2 x i32] zeroinitializer
// CK1: @__omptgt__CudaThreadsInParallel = common addrspace(3) global i32 0
// CK1: @__omptgt__SimdNumLanes = common addrspace(3) global i32 0
// CK1: @__omptgt__[[KERNUNQ:[a-zA-Z0-9_\.]+]]__thread_limit = global i32 0
// CK1: @__omptgt__[[KERNUNQ]]__simd_info = constant i8 1

int foo() {

#pragma omp target
  {
  }

  return 0;
}

// CK1: %[[PARNEST:[a-zA-Z0-9_\.]+]] = alloca i32
// CK1-NEXT: store i32 0, i32* %[[PARNEST]]
// CK1-NEXT: %[[NXTSTT:[a-zA-Z0-9_\.]+]] = alloca i32
// CK1-NEXT: store i32 0, i32* %[[NXTSTT]]
// CK1-NEXT: %[[CTLSTTIDX:[a-zA-Z0-9_\.]+]] = alloca i32
// CK1-NEXT: store i32 0, i32* %[[CTLSTTIDX]]
// CK1-NEXT: store [2 x i32] zeroinitializer, [2 x i32] addrspace(3)* @__omptgt__ControlState
// CK1-NEXT: store i32 0, i32 addrspace(3)* @__omptgt__CudaThreadsInParallel
// CK1-NEXT: %[[TID0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK1-NEXT: %[[ISMST0:[a-zA-Z0-9_\.]+]] = icmp eq i32 %[[TID0]], 0
// CK1-NEXT: br i1 %[[ISMST0]], label %[[MSTINITBLK:[a-zA-Z0-9_\.]+]], label %[[NMSTINITBLK:[a-zA-Z0-9_\.]+]]

// CK1: [[MSTINITBLK]]:
// CK1-NEXT: %[[BLKSIZE0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK1-NEXT: store i32 %[[BLKSIZE0]], i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK1-NEXT: br label %[[NMSTINITBLK]]

// CK1: [[NMSTINITBLK]]:
// CK1-NEXT: call void @llvm.nvvm.barrier0()
// CK1-NEXT: %[[FINID:[a-zA-Z0-9_\.]+]] = alloca i1
// CK1-NEXT: store i1 false, i1* %[[FINID]]
// CK1-NEXT: %[[SIMLNNM:[a-zA-Z0-9_\.]+]] = alloca i32
// CK1-DAG: %[[TID1:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK1-DAG: %[[SMDNLNS:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK1-DAG: %[[SMDNLNSM1:[a-zA-Z0-9_\.]+]] = sub i32 %[[SMDNLNS]], 1
// CK1-NEXT: %[[MYLNNM:[a-zA-Z0-9_\.]+]] = and i32 %[[TID1]], %[[SMDNLNSM1]]
// CK1-NEXT: store i32 %[[MYLNNM]], i32* %SimdLaneNum
// CK1-NEXT: br label %[[STRTCTL:[a-zA-Z0-9_\.]+]]

// CK1: [[STRTCTL]]:
// CK1-NEXT: %[[FINIDVAL:[a-zA-Z0-9_\.]+]] = load i1, i1* %[[FINID]]
// CK1-NEXT: %[[FINIDVALZERO:[a-zA-Z0-9_\.]+]] = icmp eq i1 %[[FINIDVAL]], true
// CK1-NEXT: br i1 %[[FINIDVALZERO]], label %[[ENDTGT:[a-zA-Z0-9_\.]+]], label %[[SWTCH:[a-zA-Z0-9_\.]+]]

// CK1: [[SWTCH]]:
// CK1-NEXT: %[[NXTSTTVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[NXTSTT]]
// CK1-NEXT: switch i32 %[[NXTSTTVAL]], label %[[DFLT:[a-zA-Z0-9_\.]+]] [
// CK1-NEXT: i32 0, label %[[SQSTRTCHK:[a-zA-Z0-9_\.]+]]
// CK1-NEXT: i32 1, label %[[FINIDCS:[a-zA-Z0-9_\.]+]]
// CK1-NEXT:  ]

// CK1: [[ENDTGT]]:
// CK1-NEXT: ret void

// CK1: [[SQSTRTCHK]]:
// CK1-NEXT: %[[TIDVAL:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK1-NEXT: %[[AMMST:[a-zA-Z0-9_\.]+]] = icmp eq i32 %[[TIDVAL]], 0
// CK1-NEXT: br i1 %[[AMMST]], label %[[FSTSQ:[a-zA-Z0-9_\.]+]], label %[[SYNC:[a-zA-Z0-9_\.]+]]

// CK1: [[SYNC]]:
// CK1-NEXT: call void @llvm.nvvm.barrier0()
// CK1-NEXT: %[[CTLSTTIDXVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK1-NEXT: %[[CTLSTTPOSPT:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL]]
// CK1-NEXT: %[[CTLSTTPOSVAL:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* %[[CTLSTTPOSPT]]
// CK1-NEXT: store i32 %[[CTLSTTPOSVAL]], i32* %[[NXTSTT]]
// CK1-NEXT: %[[CTLSTTIDXVAL1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK1-NEXT: %[[CTLSTTIDXVAL1XED:[a-zA-Z0-9_\.]+]] = xor i32 %[[CTLSTTIDXVAL1]], 1
// CK1-NEXT: store i32 %[[CTLSTTIDXVAL1XED]], i32* %[[CTLSTTIDX]]
// CK1-NEXT: br label %[[STRTCTL]]

// CK1: [[DFLT]]:
// CK1-NEXT: br label %[[SYNC]]

// CK1: [[FINIDCS]]:
// CK1-NEXT: store i1 true, i1* %[[FINID]]
// CK1-NEXT: br label %[[SYNC]]

// CK1: [[FSTSQ]]:
// CK1-NEXT: %[[THLIMGBL:[a-zA-Z0-9_\.]+]] = load i32, i32* @__omptgt__[[KERNUNQ]]__thread_limit
// CK1-NEXT: call void @__kmpc_kernel_init(i32 %[[THLIMGBL]])
// CK1-NEXT: %[[CTLSTTIDXVAL2:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK1-NEXT: %[[CTLSTTPOSPT1:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL2]]
// CK1-NEXT: store i32 1, i32 addrspace(3)* %[[CTLSTTPOSPT1]]
// CK1-NEXT: br label %[[SYNC]]

#endif

///##############################################
///
/// Only one declaration and initialization
///
///##############################################

#ifdef TT2
// RUN:   %clang -fopenmp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-linux \
// RUN:   -DTT2 -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK2 -input-file=target_control_loop_codegen_for_c.ll.tgt-nvptx64sm_35-nvidia-linux %s

// CK2: @__omptgt__ControlState = common addrspace(3) global [2 x i32] zeroinitializer
// CK2: @__omptgt__CudaThreadsInParallel = common addrspace(3) global i32 0
// CK2: @__omptgt__SimdNumLanes = common addrspace(3) global i32 0
// CK2: @__omptgt__[[KERNUNQ2:[a-zA-Z0-9_\.]+]]__thread_limit = global i32 0
// CK2: @__omptgt__[[KERNUNQ2]]__simd_info = constant i8 1

int foo() {

#pragma omp target
  {
    int ab = 1;
  }

  return 0;
}

// CK2: %[[AB:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK2-NEXT: %[[PARNEST:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2-NEXT: store i32 0, i32* %[[PARNEST]]
// CK2-NEXT: %[[NXTSTT:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2-NEXT: store i32 0, i32* %[[NXTSTT]]
// CK2-NEXT: %[[CTLSTTIDX:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2-NEXT: store i32 0, i32* %[[CTLSTTIDX]]
// CK2-NEXT: store [2 x i32] zeroinitializer, [2 x i32] addrspace(3)* @__omptgt__ControlState
// CK2-NEXT: store i32 0, i32 addrspace(3)* @__omptgt__CudaThreadsInParallel
// CK2-NEXT: %[[TID0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-NEXT: %[[ISMST0:[a-zA-Z0-9_\.]+]] = icmp eq i32 %[[TID0]], 0
// CK2-NEXT: br i1 %[[ISMST0]], label %[[MSTINITBLK:[a-zA-Z0-9_\.]+]], label %[[NMSTINITBLK:[a-zA-Z0-9_\.]+]]

// CK2: [[MSTINITBLK]]:
// CK2-NEXT: %[[BLKSIZE0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK2-NEXT: store i32 %[[BLKSIZE0]], i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK2-NEXT: br label %[[NMSTINITBLK]]

// CK2: [[NMSTINITBLK]]:
// CK2-NEXT: call void @llvm.nvvm.barrier0()
// CK2-NEXT: %[[FINID:[a-zA-Z0-9_\.]+]] = alloca i1
// CK2-NEXT: store i1 false, i1* %[[FINID]]
// CK2-NEXT: %[[SIMLNNM:[a-zA-Z0-9_\.]+]] = alloca i32
// CK2-DAG: %[[TID1:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-DAG: %[[SMDNLNS:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK2-DAG: %[[SMDNLNSM1:[a-zA-Z0-9_\.]+]] = sub i32 %[[SMDNLNS]], 1
// CK2-NEXT: %[[MYLNNM:[a-zA-Z0-9_\.]+]] = and i32 %[[TID1]], %[[SMDNLNSM1]]
// CK2-NEXT: store i32 %[[MYLNNM]], i32* %SimdLaneNum
// CK2-NEXT: br label %[[STRTCTL:[a-zA-Z0-9_\.]+]]

// CK2: [[STRTCTL]]:
// CK2-NEXT: %[[FINIDVAL:[a-zA-Z0-9_\.]+]] = load i1, i1* %[[FINID]]
// CK2-NEXT: %[[FINIDVALZERO:[a-zA-Z0-9_\.]+]] = icmp eq i1 %[[FINIDVAL]], true
// CK2-NEXT: br i1 %[[FINIDVALZERO]], label %[[ENDTGT:[a-zA-Z0-9_\.]+]], label %[[SWTCH:[a-zA-Z0-9_\.]+]]

// CK2: [[SWTCH]]:
// CK2-NEXT: %[[NXTSTTVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[NXTSTT]]
// CK2-NEXT: switch i32 %[[NXTSTTVAL]], label %[[DFLT:[a-zA-Z0-9_\.]+]] [
// CK2-NEXT: i32 0, label %[[SQSTRTCHK:[a-zA-Z0-9_\.]+]]
// CK2-NEXT: i32 1, label %[[FINIDCS:[a-zA-Z0-9_\.]+]]
// CK2-NEXT:  ]

// CK2: [[ENDTGT]]:
// CK2-NEXT: ret void

// CK2: [[SQSTRTCHK]]:
// CK2-NEXT: %[[TIDVAL:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK2-NEXT: %[[AMMST:[a-zA-Z0-9_\.]+]] = icmp eq i32 %[[TIDVAL]], 0
// CK2-NEXT: br i1 %[[AMMST]], label %[[FSTSQ:[a-zA-Z0-9_\.]+]], label %[[SYNC:[a-zA-Z0-9_\.]+]]

// CK2: [[SYNC]]:
// CK2-NEXT: call void @llvm.nvvm.barrier0()
// CK2-NEXT: %[[CTLSTTIDXVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK2-NEXT: %[[CTLSTTPOSPT:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL]]
// CK2-NEXT: %[[CTLSTTPOSVAL:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* %[[CTLSTTPOSPT]]
// CK2-NEXT: store i32 %[[CTLSTTPOSVAL]], i32* %[[NXTSTT]]
// CK2-NEXT: %[[CTLSTTIDXVAL1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK2-NEXT: %[[CTLSTTIDXVAL1XED:[a-zA-Z0-9_\.]+]] = xor i32 %[[CTLSTTIDXVAL1]], 1
// CK2-NEXT: store i32 %[[CTLSTTIDXVAL1XED]], i32* %[[CTLSTTIDX]]
// CK2-NEXT: br label %[[STRTCTL]]

// CK2: [[DFLT]]:
// CK2-NEXT: br label %[[SYNC]]

// CK2: [[FINIDCS]]:
// CK2-NEXT: store i1 true, i1* %[[FINID]]
// CK2-NEXT: br label %[[SYNC]]

// CK2: [[FSTSQ]]:
// CK2-NEXT: %[[THLIMGBL:[a-zA-Z0-9_\.]+]] = load i32, i32* @__omptgt__[[KERNUNQ2]]__thread_limit
// CK2-NEXT: call void @__kmpc_kernel_init(i32 %[[THLIMGBL]])
// CK2-NEXT: store i32 1, i32* %[[AB]], align 4
// CK2-NEXT: %[[CTLSTTIDXVAL2:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK2-NEXT: %[[CTLSTTPOSPT1:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL2]]
// CK2-NEXT: store i32 1, i32 addrspace(3)* %[[CTLSTTPOSPT1]]
// CK2-NEXT: br label %[[SYNC]]

#endif

///##############################################
///
/// Parallel region
///
///##############################################

#ifdef TT3
// RUN:   %clang -fopenmp -target powerpc64le-ibm-linux-gnu -omptargets=nvptx64sm_35-nvidia-linux \
// RUN:   -DTT3 -O0 -S -emit-llvm %s 2>&1
// RUN:   FileCheck -check-prefix=CK3 -input-file=target_control_loop_codegen_for_c.ll.tgt-nvptx64sm_35-nvidia-linux %s

// CK3: @__omptgt__ControlState = common addrspace(3) global [2 x i32] zeroinitializer
// CK3: @__omptgt__CudaThreadsInParallel = common addrspace(3) global i32 0
// CK3: @__omptgt__SimdNumLanes = common addrspace(3) global i32 0
// CK3: @__omptgt__[[KERNUNQ3:[a-zA-Z0-9_\.]+]]__thread_limit = global i32 0
// CK3: @__omptgt__[[KERNUNQ3]]__simd_info = constant i8 1
// CK3: @__omptgt__shared_data_ = common addrspace(3) global [{{[0-9]+}} x i8] zeroinitializer

#include <stdio.h>

int foo() {
  int b[1024];

  for (int i = 0 ; i < 1024 ; i++)
    b[i] = i;
  
#pragma omp target
  {
    int a = 1;
    
#pragma omp parallel for
    for (int i = 0 ; i < 1024 ; i++)
      b[i] += a;
  }
  
  for (int i = 0 ; i < 1024 ; i++)
    printf("b[%d] = %d\n", i, b[i]);

  return 0;
}

// CK3: %[[GBLUBSTK:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[DBGUB:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[LST:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[LB:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[UB:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[ST:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[IDX:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3-NEXT: %[[IPRIV:[a-zA-Z0-9_\.]+]] = alloca i32, align 4
// CK3: %[[PARNEST:[a-zA-Z0-9_\.]+]] = alloca i32
// CK3-NEXT: store i32 0, i32* %[[PARNEST]]
// CK3-NEXT: %[[NXTSTT:[a-zA-Z0-9\.]+]] = alloca i32
// CK3-NEXT: store i32 0, i32* %[[NXTSTT]]
// CK3-NEXT: %[[CTLSTTIDX:[a-zA-Z0-9_\.]+]] = alloca i32
// CK3-NEXT: store i32 0, i32* %[[CTLSTTIDX]]
// CK3-NEXT: store [2 x i32] zeroinitializer, [2 x i32] addrspace(3)* @__omptgt__ControlState
// CK3-NEXT: store i32 0, i32 addrspace(3)* @__omptgt__CudaThreadsInParallel
// CK3-NEXT: %[[TID0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-NEXT: %[[ISMST0:[a-zA-Z0-9_\.]+]] = icmp eq i32 %[[TID0]], 0
// CK3-NEXT: br i1 %[[ISMST0]], label %[[MSTINITBLK:[a-zA-Z0-9_\.]+]], label %[[NMSTINITBLK:[a-zA-Z0-9_\.]+]]

// CK3: [[MSTINITBLK]]:
// CK3-NEXT: %[[BLKSIZE0:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK3-NEXT: store i32 %[[BLKSIZE0]], i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-NEXT: br label %[[NMSTINITBLK]]

// CK3: [[NMSTINITBLK]]:
// CK3-NEXT: call void @llvm.nvvm.barrier0()
// CK3-NEXT: %[[FINID:[a-zA-Z0-9_\.]+]] = alloca i1
// CK3-NEXT: store i1 false, i1* %[[FINID]]
// CK3-NEXT: %[[SIMDLNNM:[a-zA-Z0-9_\.]+]] = alloca i32
// CK3-DAG: %[[TID1:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-DAG: %[[SMDNLNS:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-DAG: %[[SMDNLNSM1:[a-zA-Z0-9_\.]+]] = sub i32 %[[SMDNLNS]], 1
// CK3-NEXT: %[[MYLNNM:[a-zA-Z0-9_\.]+]] = and i32 %[[TID1]], %[[SMDNLNSM1]]
// CK3-NEXT: store i32 %[[MYLNNM]], i32* %SimdLaneNum
// CK3-NEXT: br label %[[STRTCTL:[a-zA-Z0-9_\.]+]]

// CK3: [[STRTCTL]]:
// CK3-NEXT: %[[FINIDVAL:[a-zA-Z0-9_\.]+]] = load i1, i1* %[[FINID]]
// CK3-NEXT: %[[FINIDVALZERO:[a-zA-Z0-9_\.]+]] = icmp eq i1 %[[FINIDVAL]], true
// CK3-NEXT: br i1 %[[FINIDVALZERO]], label %[[ENDTGT:[a-zA-Z0-9_\.]+]], label %[[SWTCH:[a-zA-Z0-9_\.]+]]

// CK3: [[SWTCH]]:
// CK3-NEXT: %[[NXTSTTVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[NXTSTT]]
// CK3-NEXT: switch i32 %[[NXTSTTVAL]], label %[[DFLT:[a-zA-Z0-9_\.]+]] [
// CK3-NEXT: i32 0, label %[[SQSTRTCHK:[a-zA-Z0-9_\.]+]]
// CK3-NEXT: i32 1, label %[[FINIDCS:[a-zA-Z0-9_\.]+]]
// CK3-NEXT: i32 2, label %[[PARREGPRE:[a-zA-Z0-9_\.]+]]
// CK3-NEXT: i32 3, label %[[AFTBARCHK:[a-zA-Z0-9_\.]+]]
// CK3-NEXT: i32 4, label %[[SEQREGPRE:[a-zA-Z0-9_\.]+]]
// CK3-NEXT:  ]

// CK3: [[ENDTGT]]:
// CK3-NEXT: ret void

// CK3: [[SQSTRTCHK]]:
// CK3-NEXT: %[[TIDVAL:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-NEXT: %[[AMMST:[a-zA-Z0-9_\.]+]] = icmp eq i32 %[[TIDVAL]], 0
// CK3-NEXT: br i1 %[[AMMST]], label %[[FSTSQ:[a-zA-Z0-9_\.]+]], label %[[SYNC:[a-zA-Z0-9_\.]+]]

// CK3: [[SYNC]]:
// CK3-NEXT: call void @llvm.nvvm.barrier0()
// CK3-NEXT: %[[CTLSTTIDXVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK3-NEXT: %[[CTLSTTPOSPT:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL]]
// CK3-NEXT: %[[CTLSTTPOSVAL:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* %[[CTLSTTPOSPT]]
// CK3-NEXT: store i32 %[[CTLSTTPOSVAL]], i32* %[[NXTSTT]]
// CK3-NEXT: %[[CTLSTTIDXVAL1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK3-NEXT: %[[CTLSTTIDXVAL1XED:[a-zA-Z0-9_\.]+]] = xor i32 %[[CTLSTTIDXVAL1]], 1
// CK3-NEXT: store i32 %[[CTLSTTIDXVAL1XED]], i32* %[[CTLSTTIDX]]
// CK3-NEXT: br label %[[STRTCTL]]

// CK3: [[DFLT]]:
// CK3-NEXT: br label %[[SYNC]]

// CK3: [[FINIDCS]]:
// CK3-NEXT: store i1 true, i1* %[[FINID]]
// CK3-NEXT: br label %[[SYNC]]

// CK3: [[FSTSQ]]:
// CK3-NEXT: %[[THLIMGBL:[a-zA-Z0-9_\.]+]] = load i32, i32* @__omptgt__[[KERNUNQ3]]__thread_limit
// CK3-NEXT: call void @__kmpc_kernel_init(i32 %[[THLIMGBL]])
// CK3-NEXT: store i32 1, i32* {{.*}} @__omptgt__shared_data_{{.*}} align 4
// CK3-NEXT: store i32 1, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-NEXT: %[[SMDNLNS:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-NEXT: %[[NTIDVAL:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CK3-NEXT: %[[PARTHSVAL:[a-zA-Z0-9_\.]+]] = call i32 @__kmpc_kernel_prepare_parallel(i32 %[[NTIDVAL]], i32 %[[SMDNLNS]])
// CK3-NEXT: store i32 %[[PARTHSVAL]], i32 addrspace(3)* @__omptgt__CudaThreadsInParallel
// CK3-NEXT: %[[CTLSTTIDXVAL2:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK3-NEXT: %[[CTLSTTPOSPT:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL2]]
// CK3-NEXT: store i32 2, i32 addrspace(3)* %[[CTLSTTPOSPT]]
// CK3-NEXT: br label %[[SYNC]]

// CK3: [[PARREGPRE]]:
// CK3-NEXT: %[[PARNEST_TMP1:[0-9]+]] = load i32, i32* %[[PARNEST]]
// CK3-NEXT: %[[PARNEST_TMP2:[0-9]+]] = add i32 %[[PARNEST_TMP1]], 1
// CK3-NEXT: store i32 %[[PARNEST_TMP2]], i32* %[[PARNEST]]
// CK3-NEXT: %[[TIDVAL2:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-NEXT: %[[PARTHSVAL1:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__CudaThreadsInParallel
// CK3-NEXT: %[[AMITHORLN:[a-zA-Z0-9_\.]+]] = icmp sge i32 %[[TIDVAL2]], %[[PARTHSVAL1]]
// CK3-NEXT: br i1 %[[AMITHORLN]], label %[[ISEXCL:[a-zA-Z0-9_\.]+]], label %[[ISNOTEXCL:[a-zA-Z0-9_\.]+]]

// CK3: [[ISEXCL]]:
// CK3-NEXT: %[[SMDNUMLNSVAL:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-NEXT: store i32 %[[SMDNUMLNSVAL]], i32* %[[SIMDLNNM]]
// CK3-NEXT: br label %[[SYNC]]

// CK3: [[ISNOTEXCL]]:
// CK3-DAG: %[[TIDVAL3:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-DAG: %[[SMDNUMLNSVAL1:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-DAG: %[[SMDNUMLNSVALMIN1:[a-zA-Z0-9_\.]+]] = sub i32 %[[SMDNUMLNSVAL1]], 1
// CK3-NEXT: %[[ANDTIDSMDNLNSMIN1:[a-zA-Z0-9_\.]+]] = and i32 %[[TIDVAL3]], %[[SMDNUMLNSVALMIN1]]
// CK3-NEXT: store i32 %[[ANDTIDSMDNLNSMIN1]], i32* %[[SIMDLNNM]]
// CK3-NEXT: %[[SMDNUMLNSVAL2:[a-zA-Z0-9_\.]+]] = load i32, i32 addrspace(3)* @__omptgt__SimdNumLanes
// CK3-NEXT: call void @__kmpc_kernel_parallel(i32 %[[SMDNUMLNSVAL2]])
// CK3-NEXT: %[[SIMDLNNMVAL:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[SIMDLNNM]]
// CK3-NEXT: %[[SIMDLNNMVALISZERO:[a-zA-Z0-9_\.]+]] = icmp ne i32 %[[SIMDLNNMVAL]], 0
// CK3-NEXT: br i1 %[[SIMDLNNMVALISZERO]], label %[[SYNC]], label %[[PARREGBODY:[a-zA-Z0-9_\.]+]]

// CK3: call void @__kmpc_for_static_fini({ i32, i32, i32, i32, i8* }* %[[TMP:[a-zA-Z0-9_\.]+]], i32 %[[GIDVAL:[a-zA-Z0-9_\.]+]])
// CK3-NEXT: %[[TIDVAL7:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-NEXT: %[[AMINOTMST:[a-zA-Z0-9_\.]+]] = icmp ne i32 %[[TIDVAL7]], 0
// CK3-NEXT:br i1 %[[AMINOTMST]], label %[[SYNC]], label %[[MSTNXTLBL:[a-zA-Z0-9_\.]+]]

// CK3: [[AFTBARCHK]]:
// CK3-NEXT: %[[SIMDLNNMVAL1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[SIMDLNNM]]
// CK3-NEXT: %[[SIMDLNNMVALISZERO1:[a-zA-Z0-9_\.]+]] = icmp ne i32 %[[SIMDLNNMVAL1]], 0
// CK3-NEXT: br i1 %[[SIMDLNNMVALISZERO1]], label %[[SYNC]], label %[[AFTBRRCG:[a-zA-Z0-9_\.]+]]

// CK3: [[MSTNXTLBL]]:
// CK3-NEXT: %[[CTLSTTIDX1:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK3-NEXT: %[[CTLSTTPTR1:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDX1]]
// CK3-NEXT: store i32 3, i32 addrspace(3)* %[[CTLSTTPTR1]]
// CK3-NEXT: br label %[[SYNC]]

// CK3: [[AFTBRRCG]]:
// CK3-NEXT: br label %[[LOOPPRECEND:[a-zA-Z0-9_\.]+]]

// CK3: [[LOOPPRECEND]]:
// CK3-NEXT: %[[PARNEST_TMP3:[0-9]+]] = load i32, i32* %[[PARNEST]]
// CK3-NEXT: %[[PARNEST_TMP4:[0-9]+]] = sub i32 %[[PARNEST_TMP3]], 1
// CK3-NEXT: store i32 %[[PARNEST_TMP4]], i32* %[[PARNEST]]
// CK3-NEXT: call void @__kmpc_kernel_end_parallel()
// CK3-NEXT: %[[TIDVAL8:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-NEXT: %[[AMINOTMST1:[a-zA-Z0-9_\.]+]] = icmp ne i32 %[[TIDVAL8]], 0
// CK3-NEXT: br i1 %[[AMINOTMST1]], label %[[SYNC]], label %[[MSTNXTLBL1:[a-zA-Z0-9_\.]+]]

// CK3: [[SEQREGPRE]]:
// CK3-NEXT: %[[TIDVAL9:[a-zA-Z0-9_\.]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CK3-NEXT: %[[AMINOTMST2:[a-zA-Z0-9_\.]+]] = icmp ne i32 %[[TIDVAL9]], 0
// CK3-NEXT: br i1 %[[AMINOTMST2]], label %[[SYNC]], label %[[MSTONLYSEQREG:[a-zA-Z0-9_\.]+]]

// CK3: [[MSTNXTLBL1]]:
// CK3-NEXT: %[[CTLSTTIDX2:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK3-NEXT: %[[CTLSTTPTR2:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDX2]]
// CK3-NEXT: store i32 4, i32 addrspace(3)* %[[CTLSTTPTR2]]
// CK3-NEXT: br label %[[SYNC]]

// CK3: [[MSTONLYSEQREG]]:
// CK3-NEXT: %[[CTLSTTIDXVAL2:[a-zA-Z0-9_\.]+]] = load i32, i32* %[[CTLSTTIDX]]
// CK3-NEXT: %[[CTLSTTPOSPT1:[a-zA-Z0-9_\.]+]] = getelementptr [2 x i32], [2 x i32] addrspace(3)* @__omptgt__ControlState, i32 0, i32 %[[CTLSTTIDXVAL2]]
// CK3-NEXT: store i32 1, i32 addrspace(3)* %[[CTLSTTPOSPT1]]
// CK3-NEXT: br label %[[SYNC]]

#endif
