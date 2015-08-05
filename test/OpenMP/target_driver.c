///
/// Perform several driver tests for OpenMP offloading
///

/// Check whether an invalid OpenMP target is specified:
// RUN:   %clang -### -fopenmp -omptargets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'

/// Check error for empty -omptargets
// RUN:   %clang -### -fopenmp -omptargets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-OMPTARGETS %s
// CHK-EMPTY-OMPTARGETS: warning: joined argument expects additional value: '-omptargets='

/// Check whether we are using a target whose toolchain was not prepared to
/// to support offloading:
// RUN:   %clang -### -fopenmp -omptargets=x86_64-apple-darwin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SUPPORT %s
// CHK-NO-SUPPORT: error: Toolchain for target 'x86_64-apple-darwin' is not supporting OpenMP offloading.

/// Target independent check of the commands passed to each tool when using
/// valid OpenMP targets
// RUN:   %clang -### -fopenmp -target powerpc64-linux -omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-COMMANDS %s
//

// Host commands: PPC uses integrated preprocessor and assembler.
// However, given that we have OpenMP targets the compile and backend phases are not combined.
// CHK-COMMANDS: clang{{.*}}" "-cc1"
// CHK-COMMANDS: "-emit-llvm-bc"
// CHK-COMMANDS: "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-main-file-path" "[[SRC:[^ ]+]].c"
// CHK-COMMANDS: "-o" "[[HOSTBC:.+]].bc"
// CHK-COMMANDS: "-x" "c" "[[SRC]].c"

// CHK-COMMANDS: clang{{.*}}" "-cc1"
// CHK-COMMANDS: "-emit-obj"
// CHK-COMMANDS: "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[HOSTOBJ:.+]].o"
// CHK-COMMANDS: "-x" "ir" "[[HOSTBC]].bc"

// Target1 command PPC uses integrated assembler. Preprocessor is not integrated
// as the compiler phase needs to take the *.bc file from the host.
// CHK-COMMANDS: clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-linux-gnu" "-E" {{.*}}"-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[T1PP:.+]].i" "-x" "c" "[[SRC]].c"
// CHK-COMMANDS: clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-linux-gnu" "-emit-llvm-bc" {{.*}}"-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-omp-host-output-file-path" "[[HOSTBC]].bc" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[T1BC:.+]].bc" "-x" "cpp-output" "[[T1PP]].i"
// CHK-COMMANDS: clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-linux-gnu" "-emit-obj" {{.*}}"-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[T1OBJ:.+]].o" "-x" "ir" "[[T1BC]].bc"
// CHK-COMMANDS: ld" {{.*}}"--eh-frame-hdr" "-m" "elf64ppc" "-shared" "-o" "[[T1LIB:.+]].so" {{.*}} "[[T1OBJ]].o"

// Target2 commands (nvptx uses host preprocessor definitions)
// CHK-COMMANDS: clang{{.*}}" "-cc1" "-triple" "powerpc64--linux" "-E" {{.*}}"-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[T2PP:.+]].i" "-x" "c" "[[SRC]].c"
// CHK-COMMANDS: clang{{.*}}" "-cc1" "-triple" "nvptx64-nvidia-cuda" "-emit-llvm-bc" {{.*}}"-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-omp-host-output-file-path" "[[HOSTBC]].bc" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[T2BC:.+]].bc" "-x" "cpp-output" "[[T2PP]].i"
// CHK-COMMANDS: clang{{.*}}" "-cc1" "-triple" "nvptx64-nvidia-cuda" "-S"
// CHK-COMMANDS: "-target-cpu" "sm_20"
// CHK-COMMANDS: "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-omp-main-file-path" "[[SRC]].c"
// CHK-COMMANDS: "-o" "[[T2ASM:.+]].s" "-x" "ir" "[[T2BC]].bc"
// CHK-COMMANDS: ptxas" "-o" "[[T2OBJ:.+]].o" "-c" "-arch" "sm_20" "-maxrregcount" "64" "[[T2ASM]].s"
// CHK-COMMANDS: cp" "[[T2OBJ]].o" "[[T2CBIN:.+]].cubin"
// CHK-COMMANDS: nvlink" "-o" "[[T2LIB:.+]].so" "-arch" "sm_20" {{.*}}"[[T2CBIN]].cubin"

// Final linking command
// CHK-COMMANDS: ld" {{.*}}"-o" "a.out"  {{.*}}"[[HOSTOBJ]].o" "-lomp" "-lomptarget" {{.*}} "-T" "[[LKSCRIPT:.+]].lk"

/// Check frontend require main file name
// RUN:   not %clang_cc1 "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-triple" "powerpc64-ibm-linux-gnu" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MAINFILE %s
// RUN:   not %clang_cc1 "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-triple" "nvptx64-nvidia-cuda" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MAINFILE %s
// CHK-MAINFILE: error: Main-file path is required to generate code for OpenMP target regions. Use -omp-main-file-path 'path'.

/// Check frontend module ID error
// RUN:   not %clang_cc1 "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-triple" "powerpc64-ibm-linux-gnu" "-omp-main-file-path" "abcd.efgh" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MODULEID %s
// RUN:   not %clang_cc1 "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-omp-target-mode" "-triple" "nvptx64-nvidia-cuda" %s "-omp-main-file-path" "abcd.efgh" 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-MODULEID %s
// CHK-MODULEID: error: Unable to generate module ID from input file 'abcd.efgh' for OpenMP target code generation. Make sure the file exists in the file system.

/// Check the subtarget detection
// RUN:   %clang -### -fopenmp -target powerpc64-linux -omptargets=nvptx64sm_35-nvidia-cuda %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SUBTARGET %s
// CHK-SUBTARGET: clang{{.*}}" "-cc1" "-triple" "nvptx64sm_35-nvidia-cuda" "-S"
// CHK-SUBTARGET: "-target-cpu" "sm_35"
// CHK-SUBTARGET: "-fopenmp" "-omptargets=nvptx64sm_35-nvidia-cuda" "-omp-target-mode" "-omp-main-file-path"
// CHK-SUBTARGET: "-o" "[[ASM:.+]].s"
// CHK-SUBTARGET: "-x" "ir" "[[BC:.+]].bc"

/// Check the automatic detection of target files
// RUN:   %clang -fopenmp -target powerpc64-linux -omptargets=nvptx64sm_35-nvidia-cuda %s -S 2>&1
// RUN:   %clang -### -fopenmp -target powerpc64-linux -omptargets=nvptx64sm_35-nvidia-cuda target_driver.s -Womp-implicit-target-files 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-WARN-IMPLICIT %s
// RUN:   %clang -### -fopenmp -target powerpc64-linux -omptargets=nvptx64sm_35-nvidia-cuda target_driver.s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TARGET-NOTWARN-IMPLICIT %s
// CHK-TARGET-WARN-IMPLICIT: warning: OpenMP target file 'target_driver.s.tgt-nvptx64sm_35-nvidia-cuda' is being implicitly used in the 'nvptx64sm_35-nvidia-cuda' toolchain.
// CHK-TARGET-NOTWARN-IMPLICIT-NOT: warning: OpenMP target file 'target_driver.s.tgt-nvptx64sm_35-nvidia-cuda' is being implicitly used in the 'nvptx64sm_35-nvidia-cuda' toolchain.

/// Check separate compilation
// RUN:   echo ' ' > %t.1.s 
// RUN:   echo ' ' > %t.1.s.tgt-nvptx64sm_35-nvidia-cuda
// RUN:   echo ' ' > %t.2.o
// RUN:   %clang -### -fopenmp -target powerpc64-linux -omptargets=nvptx64sm_35-nvidia-cuda %t.1.s %t.2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SEP-COMPILATION %s
// CHK-SEP-COMPILATION: clang{{.*}}" "-cc1as" "-triple" "powerpc64--linux" "-filetype" "obj" {{.*}}"-o" "[[HOSTOBJ:.+]].o" "[[HOSTASM:.+]].s"
// CHK-SEP-COMPILATION: ptxas" "-o" "[[TGTOBJ:.+]].o" "-c" "-arch" "sm_35" "-maxrregcount" "64" "[[HOSTASM]].s.tgt-nvptx64sm_35-nvidia-cuda"
// CHK-SEP-COMPILATION: {{.*}}" "[[TGTOBJ]].o" "[[TGTCUBIN:.+]].cubin"
// CHK-SEP-COMPILATION: nvlink" "-o" "[[TGTSO:.+]].so"  {{.*}}"[[TGTCUBIN]].cubin"
// CHK-SEP-COMPILATION: ld" {{.*}}"[[HOSTOBJ]].o" "[[HOSTOBJ2:.+]].o" {{.*}}"-T" "[[LKS:.+]].lk"

/// Check host output file no exist message
// RUN:   not %clang_cc1 "-fopenmp" "-omptargets=powerpc64-ibm-linux-gnu,nvptx64-nvidia-cuda" "-triple" "powerpc64-ibm-linux-gnu" "-omp-host-output-file-path" "abcd.efgh" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-HOSTFILE-NOEXIST %s
// CHK-HOSTFILE-NOEXIST: error: The provided  host compiler output 'abcd.efgh' is required to generate code for OpenMP target regions and cannot be found.

/// Check error if libdevice is not found
// RUN:   not %clang -fopenmp -target powerpc64-linux -omptargets=nvptx64-nvidia-cuda %s -lm 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LIBDEVICE %s
// CHK-LIBDEVICE: error: CUDA math library (libdevice) is required and cannot be found.

/// Check error if something wrong with nvptx sharing stack options
// RUN:   not %clang -fopenmp -target powerpc64le-linux -omptargets=nvptx64-nvidia-cuda \
// RUN:   -omp-nvptx-data-sharing-type=bla \
// RUN:   -omp-nvptx-data-sharing-sizes-per-thread=64,x24 \
// RUN:   -omp-nvptx-data-sharing-size-per-team=123x45 \
// RUN:   -omp-nvptx-data-sharing-size-per-kernel=678x90 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NVPTX-SHARING %s
// CHK-NVPTX-SHARING-DAG: error: invalid value 'bla' in '-omp-nvptx-data-sharing-type=bla'
// CHK-NVPTX-SHARING-DAG: error: invalid value 'x24' in '-omp-nvptx-data-sharing-sizes-per-thread=64,x24'
// CHK-NVPTX-SHARING-DAG: error: invalid value '123x45' in '-omp-nvptx-data-sharing-size-per-team=123x45'
// CHK-NVPTX-SHARING-DAG: error: invalid value '678x90' in '-omp-nvptx-data-sharing-size-per-kernel=678x90'

