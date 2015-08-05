//===----- CGOpenMPRuntime.h - Interface to OpenMP Runtimes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenMP code generation.  Concrete
// subclasses of this implement code generation for specific OpenMP
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIME_H

#include "clang/AST/Type.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"

namespace llvm {
class AllocaInst;
class CallInst;
class GlobalVariable;
class Constant;
class Function;
class Module;
class StructLayout;
class FunctionType;
class StructType;
class Type;
class Value;
} // namespace llvm

namespace clang {

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

#define DEFAULT_EMIT_OPENMP_DECL(name)       \
  virtual llvm::Constant* Get_##name();

/// Implements runtime-specific code generation functions.
class CGOpenMPRuntime {
public:
  /// \brief Values for bit flags used in the ident_t to describe the fields.
  /// All enumeric elements are named and described in accordance with the code
  /// from http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
  enum OpenMPLocationFlags {
    /// \brief Use trampoline for internal microtask.
    OMP_IDENT_IMD = 0x01,
    /// \brief Use c-style ident structure.
    OMP_IDENT_KMPC = 0x02,
    /// \brief Atomic reduction option for kmpc_reduce.
    OMP_ATOMIC_REDUCE = 0x10,
    /// \brief Explicit 'barrier' directive.
    OMP_IDENT_BARRIER_EXPL = 0x20,
    /// \brief Implicit barrier in code.
    OMP_IDENT_BARRIER_IMPL = 0x40,
    /// \brief Implicit barrier in 'for' directive.
    OMP_IDENT_BARRIER_IMPL_FOR = 0x40,
    /// \brief Implicit barrier in 'sections' directive.
    OMP_IDENT_BARRIER_IMPL_SECTIONS = 0xC0,
    /// \brief Implicit barrier in 'single' directive.
    OMP_IDENT_BARRIER_IMPL_SINGLE = 0x140
  };
  enum OpenMPRTLFunction {
    // Call to void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro
    // microtask, ...);
    OMPRTL__kmpc_fork_call,
    // Call to kmp_int32 kmpc_global_thread_num(ident_t *loc);
    OMPRTL__kmpc_global_thread_num
  };

  enum OpenMPReservedDeviceID {
    // Device ID if the device was not defined, runtime should get it from the
    // global variables in the spec.
    OMPRTL__target_device_id_undef = -1,
    // Means target all devices and should be run the first time they hit a
    // regular target region - used for Ctors
    OMPRTL__target_device_id_ctors = -2,
    // Means target all devices and should run on all devices that were used in
    // the current shared library - used for Dtors
    OMPRTL__target_device_id_dtors = -3
  };

protected:
  CodeGenModule &CGM;
  /// \brief Default const ident_t object used for initialization of all other
  /// ident_t objects.
  llvm::Constant *DefaultOpenMPPSource;
  /// \brief Map of flags and corrsponding default locations.
  typedef llvm::DenseMap<unsigned, llvm::Value *> OpenMPDefaultLocMapTy;
  OpenMPDefaultLocMapTy OpenMPDefaultLocMap;
  llvm::Value *GetOrCreateDefaultOpenMPLocation(OpenMPLocationFlags Flags);
  /// \brief Describes ident structure that describes a source location.
  /// All descriptions are taken from
  /// http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
  /// Original structure:
  /// typedef struct ident {
  ///    kmp_int32 reserved_1;   /**<  might be used in Fortran;
  ///                                  see above  */
  ///    kmp_int32 flags;        /**<  also f.flags; KMP_IDENT_xxx flags;
  ///                                  KMP_IDENT_KMPC identifies this union
  ///                                  member  */
  ///    kmp_int32 reserved_2;   /**<  not really used in Fortran any more;
  ///                                  see above */
  ///#if USE_ITT_BUILD
  ///                            /*  but currently used for storing
  ///                                region-specific ITT */
  ///                            /*  contextual information. */
  ///#endif /* USE_ITT_BUILD */
  ///    kmp_int32 reserved_3;   /**< source[4] in Fortran, do not use for
  ///                                 C++  */
  ///    char const *psource;    /**< String describing the source location.
  ///                            The string is composed of semi-colon separated
  //                             fields which describe the source file,
  ///                            the function and a pair of line numbers that
  ///                            delimit the construct.
  ///                             */
  /// } ident_t;
  enum IdentFieldIndex {
    /// \brief might be used in Fortran
    IdentField_Reserved_1,
    /// \brief OMP_IDENT_xxx flags; OMP_IDENT_KMPC identifies this union member.
    IdentField_Flags,
    /// \brief Not really used in Fortran any more
    IdentField_Reserved_2,
    /// \brief Source[4] in Fortran, do not use for C++
    IdentField_Reserved_3,
    /// \brief String describing the source location. The string is composed of
    /// semi-colon separated fields which describe the source file, the function
    /// and a pair of line numbers that delimit the construct.
    IdentField_PSource
  };
  llvm::StructType *IdentTy;
  /// \brief Map for Sourcelocation and OpenMP runtime library debug locations.
  typedef llvm::DenseMap<unsigned, llvm::Value *> OpenMPDebugLocMapTy;
  OpenMPDebugLocMapTy OpenMPDebugLocMap;
  /// \brief The type for a microtask which gets passed to __kmpc_fork_call().
  /// Original representation is:
  /// typedef void (kmpc_micro)(kmp_int32 global_tid, kmp_int32 bound_tid,...);
  llvm::FunctionType *Kmpc_MicroTy;
  /// \brief Map of local debug location and functions.
  typedef llvm::DenseMap<llvm::Function *, llvm::Value *> OpenMPLocMapTy;
  OpenMPLocMapTy OpenMPLocMap;
  /// \brief Map of local gtid and functions.
  typedef llvm::DenseMap<llvm::Function *, llvm::Value *> OpenMPGtidMapTy;
  OpenMPGtidMapTy OpenMPGtidMap;

  // Number of target regions processed so far
  unsigned NumTargetRegions;

  // Number of globals processed so far that are to be mapped into a target
  unsigned NumTargetGlobals;

  // Name of the current function whose target regions are being identified
  std::string CurTargetParentFunctionName;

  // Set of all local variables that need to be turned global due to data sharing
  // constraints. This is organized as vector of trees.
  typedef llvm::SmallSet<llvm::Value *, 32> SharedValuesSetTy;
  typedef llvm::SmallVector<SharedValuesSetTy, 4> SharedValuesPerLevelTy;
  typedef llvm::SmallVector<SharedValuesPerLevelTy, 4> SharedValuesPerRegionTy;
  typedef llvm::SmallVector<SharedValuesPerRegionTy, 4> SharedValuesTy;
  SharedValuesTy ValuesToBeInSharedMemory;

  // Set of all global initializers required in the declare target regions
  llvm::SmallSet<const llvm::Constant*, 32> TargetGlobalInitializers;

  // Arrays that keep the order of the target global variables and entry points
  // so that the corresponding metadata can be generated when the module is
  // closed.

  enum {
    OMPTGT_METADATA_TY_GLOBAL_VAR = 0,
    OMPTGT_METADATA_TY_TARGET_REGION,
    OMPTGT_METADATA_TY_CTOR,
    OMPTGT_METADATA_TY_DTOR,
    OMPTGT_METADATA_TY_OTHER_GLOBAL_VAR,
    OMPTGT_METADATA_TY_OTHER_FUNCTION
  };

  // Global variables: {Name, order}
  typedef llvm::StringMap<unsigned> GlobalsOrderTy;
  GlobalsOrderTy GlobalsOrder;

  // Target regions: {Parent Function Name, order1, order2, ... }
  typedef llvm::SmallVector<unsigned, 32> TargetRegionOrderPerFunctionTy;
  typedef llvm::StringMap<TargetRegionOrderPerFunctionTy> TargetRegionsOrderTy;
  TargetRegionsOrderTy TargetRegionsOrder;

  // Ctor regions: {order_priority1, order_priority2, ...,
  // order_priority_default }
  typedef llvm::SmallVector<unsigned, 32> CtorRegionsOrderTy;
  CtorRegionsOrderTy CtorRegionsOrder;

  // Dtor regions: {Name being destroyed, order }
  typedef llvm::StringMap<unsigned> DtorRegionsOrderTy;
  DtorRegionsOrderTy DtorRegionsOrder;

  // Other global variables: {Name}
  typedef std::set<std::string> OtherGlobalVariablesTy;
  OtherGlobalVariablesTy OtherGlobalVariables;

  // Other funtions: {Name}
  typedef std::set<std::string> OtherFunctionsTy;
  OtherFunctionsTy OtherFunctions;

  // True if any target information was loaded from metadata
  bool HasTargetInfoLoaded;

  // Array containing the target entries, in the order they should appear
  llvm::DenseMap<llvm::GlobalVariable *, unsigned> OrderForEntry;

  // Map between declarations and the target constants.
  // This is useful to trace whether a given target region was processed before
  // and avoid duplicate the code generation. The code generation for a target
  // directive may be called more than once if ,e.g., if-clauses are employed
  typedef llvm::DenseMap<const Decl *, llvm::Constant *> DeclsToEntriesMapTy;
  DeclsToEntriesMapTy DeclsToEntriesMap;

  // Target regions descriptor for the current compilation unit
  llvm::Constant *TargetRegionsDescriptor;

  /// \brief  Return host pointer for the current target regions. This creates
  /// the offload entry for the target region. This takes the record decl
  /// associated with the target region to determine the ordering of the entry.
  ///
  virtual llvm::GlobalVariable *
  CreateHostPtrForCurrentTargetRegion(const Decl *D, llvm::Function *Fn,
                                      StringRef Name);

  /// \brief  Creates the host entry for a given global and places it in the
  /// entries reserved section
  ///
  virtual llvm::GlobalVariable *
  CreateHostEntryForTargetGlobal(const Decl *D, llvm::GlobalVariable *GV,
                                 StringRef Name);

public:
  // register the name of the current function whose target regions are being
  // identified
  void registerCurTargetParentFunctionName(StringRef s) {
    CurTargetParentFunctionName = s;
  }

  // hooks to register information that should match between host and target
  void registerGlobalVariable(const Decl *D, llvm::GlobalVariable *GV);
  void registerTargetRegion(const Decl *D, llvm::Function *Fn,
                            llvm::Function *ParentFunction);
  virtual void registerCtorRegion(llvm::Function *Fn);
  virtual void registerDtorRegion(llvm::Function *Fn,
                                  llvm::Constant *Destructee);
  void registerOtherGlobalVariable(const VarDecl *Other);
  void registerOtherFunction(const FunctionDecl *Other, StringRef Name);

  // Return true if there is any OpenMP target code to be generated
  bool hasAnyTargetCodeToBeEmitted();

  // Return true if the given name maps to any valid target global variable
  // (entry point or not
  bool isValidAnyTargetGlobalVariable(StringRef name);
  bool isValidAnyTargetGlobalVariable(const Decl *D);

  // Return true if the given name maps to a valid target global variable that
  // is also an entry point
  bool isValidEntryTargetGlobalVariable(StringRef name);

  // Return true if the given name maps to a function that contains target
  // regions that should be emitted
  bool isValidTargetRegionParent(StringRef name);

  // Return true if the given name maps to a target global variable that is
  // not an entry point
  bool isValidOtherTargetGlobalVariable(StringRef name);

  // Return true if the given name maps to a target function that is not an
  // entry point
  bool isValidOtherTargetFunction(StringRef name);

  // Return true if the current module requires a the target descriptor to be
  // registered
  bool requiresTargetDescriptorRegistry(){
    return NumTargetRegions != 0 || !TargetGlobalInitializers.empty();
  }

  // Register global initializer for OpenMP Target offloading
  void registerTargetGlobalInitializer(const llvm::Constant *D);

  // Return true if D is a global initializer for OpenMP Target offloading
  bool isTargetGlobalInitializer(const llvm::Constant *D);

  // Return true if the current module has global initializers
  bool hasTargetGlobalInitializers();

  // Start sharing region. This will initialize a new set of shared variables
  void startSharedRegion(unsigned NestingLevel);

  // Mark value as requiring to be moved to global memory
  void addToSharedRegion(llvm::Value *V, unsigned NestingLevel);

  // Return the registered constant for a given declaration
  llvm::Constant *getEntryForDeclaration(const Decl *D);

  // Register a declaration with a constant that describes the entry point
  void registerEntryForDeclaration(const Decl *D, llvm::Constant *C);

  enum EAtomicOperation {
    OMP_Atomic_add,
    OMP_Atomic_sub,
    OMP_Atomic_mul,
    OMP_Atomic_div,
    OMP_Atomic_andb,
    OMP_Atomic_shl,
    OMP_Atomic_shr,
    OMP_Atomic_orb,
    OMP_Atomic_xor,
    OMP_Atomic_andl,
    OMP_Atomic_orl,
    OMP_Atomic_max,
    OMP_Atomic_min,
    OMP_Atomic_eqv,
    OMP_Atomic_neqv,
    OMP_Atomic_rd,
    OMP_Atomic_wr,
    OMP_Atomic_swp,
    OMP_Atomic_assign,
    OMP_Atomic_invalid
  };
  
  explicit CGOpenMPRuntime(CodeGenModule &CGM);
  virtual ~CGOpenMPRuntime();

  /// \brief Cleans up references to the objects in finished function.
  /// \param CGF Reference to finished CodeGenFunction.
  ///
  void FunctionFinished(CodeGenFunction &CGF);

  /// \brief Emits object of ident_t type with info for source location.
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param Flags Flags for OpenMP location.
  ///
  llvm::Value *
  EmitOpenMPUpdateLocation(CodeGenFunction &CGF, SourceLocation Loc,
                           OpenMPLocationFlags Flags = OMP_IDENT_KMPC);

  /// \brief Generates global thread number value.
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  ///
  llvm::Value *GetOpenMPGlobalThreadNum(CodeGenFunction &CGF,
                                        SourceLocation Loc);

  /// \brief Returns pointer to ident_t type;
  llvm::Type *getIdentTyPointerTy();

  /// \brief Returns pointer to kmpc_micro type;
  llvm::Type *getKmpc_MicroPointerTy();

  /// \brief Returns specified OpenMP runtime function.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *CreateRuntimeFunction(OpenMPRTLFunction Function);

  DEFAULT_EMIT_OPENMP_DECL(fork_call)
  DEFAULT_EMIT_OPENMP_DECL(push_num_threads)
  DEFAULT_EMIT_OPENMP_DECL(push_proc_bind)
  DEFAULT_EMIT_OPENMP_DECL(fork_teams)
  DEFAULT_EMIT_OPENMP_DECL(push_num_teams)
  DEFAULT_EMIT_OPENMP_DECL(cancel_barrier)
  DEFAULT_EMIT_OPENMP_DECL(barrier)
  DEFAULT_EMIT_OPENMP_DECL(cancellationpoint)
  DEFAULT_EMIT_OPENMP_DECL(cancel)
  DEFAULT_EMIT_OPENMP_DECL(omp_taskyield)
  DEFAULT_EMIT_OPENMP_DECL(omp_taskwait)
  DEFAULT_EMIT_OPENMP_DECL(flush)
  DEFAULT_EMIT_OPENMP_DECL(master)
  DEFAULT_EMIT_OPENMP_DECL(end_master)
  DEFAULT_EMIT_OPENMP_DECL(single)
  DEFAULT_EMIT_OPENMP_DECL(end_single)
  DEFAULT_EMIT_OPENMP_DECL(critical)
  DEFAULT_EMIT_OPENMP_DECL(end_critical)
  DEFAULT_EMIT_OPENMP_DECL(ordered)
  DEFAULT_EMIT_OPENMP_DECL(end_ordered)
  DEFAULT_EMIT_OPENMP_DECL(end_reduce_nowait)
  DEFAULT_EMIT_OPENMP_DECL(end_reduce)
  DEFAULT_EMIT_OPENMP_DECL(atomic_start)
  DEFAULT_EMIT_OPENMP_DECL(atomic_end)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_init_4)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_init_4u)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_init_8)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_init_8u)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_next_4)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_next_4u)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_next_8)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_next_8u)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_fini_4)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_fini_4u)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_fini_8)
  DEFAULT_EMIT_OPENMP_DECL(dispatch_fini_8u)
  DEFAULT_EMIT_OPENMP_DECL(for_static_init_4)
  DEFAULT_EMIT_OPENMP_DECL(for_static_init_4u)
  DEFAULT_EMIT_OPENMP_DECL(for_static_init_8)
  DEFAULT_EMIT_OPENMP_DECL(for_static_init_8u)
  DEFAULT_EMIT_OPENMP_DECL(for_static_fini)
  DEFAULT_EMIT_OPENMP_DECL(omp_task_begin_if0)
  DEFAULT_EMIT_OPENMP_DECL(omp_task_complete_if0)
  DEFAULT_EMIT_OPENMP_DECL(omp_task_parts)
  DEFAULT_EMIT_OPENMP_DECL(taskgroup)
  DEFAULT_EMIT_OPENMP_DECL(end_taskgroup)
  DEFAULT_EMIT_OPENMP_DECL(register_lib)
  DEFAULT_EMIT_OPENMP_DECL(unregister_lib)

  DEFAULT_EMIT_OPENMP_DECL(threadprivate_register)
  DEFAULT_EMIT_OPENMP_DECL(global_thread_num)

  DEFAULT_EMIT_OPENMP_DECL(kernel_init)
  DEFAULT_EMIT_OPENMP_DECL(kernel_prepare_parallel)
  DEFAULT_EMIT_OPENMP_DECL(kernel_parallel)
  DEFAULT_EMIT_OPENMP_DECL(kernel_end_parallel)

  DEFAULT_EMIT_OPENMP_DECL(serialized_parallel)
  DEFAULT_EMIT_OPENMP_DECL(end_serialized_parallel)

  virtual llvm::Type *getKMPDependInfoType();

  // Special processing for __kmpc_copyprivate
  // DEFAULT_GET_OPENMP_FUNC(copyprivate)
  virtual llvm::Constant *Get_copyprivate();
  // Special processing for __kmpc_reduce_nowait
  // DEFAULT_GET_OPENMP_FUNC(reduce_nowait)
  virtual llvm::Constant * Get_reduce_nowait();
  // Special processing for __kmpc_reduce
  // DEFAULT_GET_OPENMP_FUNC(reduce)
  virtual llvm::Constant *Get_reduce();
  // Special processing for __kmpc_omp_task_alloc
  // DEFAULT_GET_OPENMP_FUNC(omp_task_alloc)
  virtual llvm::Constant * Get_omp_task_alloc();
  // Special processing for __kmpc_omp_task_with_deps
  // DEFAULT_GET_OPENMP_FUNC(omp_task_with_deps)
  virtual llvm::Constant * Get_omp_task_with_deps();
  // Special processing for __kmpc_omp_wait_deps
  // DEFAULT_GET_OPENMP_FUNC(omp_wait_deps)
  virtual llvm::Constant * Get_omp_wait_deps();

  // Special processing for __tgt_target
  virtual llvm::Constant * Get_target();
  // Special processing for __tgt_target_nowait
  virtual llvm::Constant * Get_target_nowait();
  // Special processing for __tgt_target_teams
  virtual llvm::Constant * Get_target_teams();
  // Special processing for __tgt_target_teams_nowait
  virtual llvm::Constant * Get_target_teams_nowait();
  // Special processing for __tgt_target_data_begin
  virtual llvm::Constant * Get_target_data_begin();
  // Special processing for __tgt_target_data_begin
  virtual llvm::Constant * Get_target_data_begin_nowait();
  // Special processing for __tgt_target_data_end
  virtual llvm::Constant * Get_target_data_end();
  // Special processing for __tgt_target_data_end
  virtual llvm::Constant * Get_target_data_end_nowait();
  // Special processing for __tgt_target_data_update
  virtual llvm::Constant * Get_target_data_update();
  // Special processing for __tgt_target_data_update
  virtual llvm::Constant * Get_target_data_update_nowait();

  // Special processing for __kmpc_threadprivate_cached
  // DEFAULT_GET_OPENMP_FUNC(threadprivate_cached)
  virtual llvm::Constant *  Get_threadprivate_cached();


  virtual QualType GetAtomicType(CodeGenFunction &CGF, QualType QTy);
  virtual llvm::Value *GetAtomicFuncGeneral(CodeGenFunction &CGF, QualType QTyRes,
                                           QualType QTyIn, EAtomicOperation Aop,
                                           bool Capture, bool Reverse);
  virtual llvm::Value *GetAtomicFunc(CodeGenFunction &CGF, QualType QTy,
      OpenMPReductionClauseOperator Op);

  /// Return reduction call to perform specialized reduction in a single OpenMP
  /// team if the target can benefit from it.
  virtual llvm::Value *GetTeamReduFunc(CodeGenFunction &CGF, QualType QTy,
                                       OpenMPReductionClauseOperator Op);

  /// This is a hook to enable postprocessing of the module.
  virtual void PostProcessModule(CodeGenModule &CGM);

  /// Implement some target dependent transformation for the target region
  /// outlined function
  ///
  virtual void PostProcessTargetFunction(const Decl *D,
                                          llvm::Function *F,
                                          const CGFunctionInfo &FI);
  virtual void PostProcessTargetFunction(llvm::Function *F);

  /// \brief Creates a structure with the location info for Intel OpenMP RTL.
  virtual llvm::Value *CreateIntelOpenMPRTLLoc(SourceLocation Loc,
      CodeGenFunction &CGF, unsigned Flags = 0x02);
  /// \brief Creates call to "__kmpc_global_thread_num(ident_t *loc)" OpenMP
  /// RTL function.
  virtual llvm::Value *CreateOpenMPGlobalThreadNum(SourceLocation Loc,
      CodeGenFunction &CGF);
  /// \brief Checks if the variable is OpenMP threadprivate and generates code
  /// for threadprivate variables.
  /// \return 0 if the variable is not threadprivate, or new address otherwise.
  virtual llvm::Value *CreateOpenMPThreadPrivateCached(const VarDecl *VD,
                                               SourceLocation Loc,
                                               CodeGenFunction &CGF,
                                               bool NoCast = false);

  /// \brief  Return a string with the mangled name of a target region or global
  /// entry point. The client can choose to invalidate the used order entry.
  ///
  std::string GetOffloadEntryMangledName();
  std::string GetOffloadEntryMangledName(unsigned ID);
  std::string
  GetOffloadEntryMangledNameForGlobalVariable(StringRef Key, unsigned &Order,
                                              bool Invalidate = false);
  std::string
  GetOffloadEntryMangledNameForGlobalVariable(StringRef Key,
                                              bool Invalidate = false);
  std::string
  GetOffloadEntryMangledNameForTargetRegion(unsigned &Order,
                                            bool Invalidate = false);
  std::string
  GetOffloadEntryMangledNameForTargetRegion(bool Invalidate = false);
  std::string GetOffloadEntryMangledNameForCtor(unsigned &Order,
                                                bool Invalidate = false);
  std::string GetOffloadEntryMangledNameForCtor(bool Invalidate = false);
  std::string GetOffloadEntryMangledNameForDtor(StringRef Key, unsigned &Order,
                                                bool Invalidate = false);
  std::string GetOffloadEntryMangledNameForDtor(StringRef Key,
                                                bool Invalidate = false);

  /// \brief  Return the target regions descriptor or a create a new
  /// one if if does not exist
  ///
  llvm::Constant* GetTargetRegionsDescriptor();

  /// \brief Return a pointer to the device image begin
  ///
  llvm::Constant* GetDeviceImageBeginPointer(llvm::Triple TargetTriple);

  /// \brief Return a pointer to the device image end
  ///
  llvm::Constant* GetDeviceImageEndPointer(llvm::Triple TargetTriple);

  // \brief If needed, re-initialize part of the state
  virtual void StartNewTargetRegion();

  // \brief if needed, record that codegen hit teams construct
  virtual void StartTeamsRegion();

  // \brief in default case, just emit call to omp RT barrier
  // in other cases more codegen may be needed
  virtual void EmitOMPBarrier(SourceLocation L, unsigned Flags,
                              CodeGenFunction &CGF);

  /// \brief Code generation helper in target regions. Create a control-loop
  //  with inspector/executor for special back-ends (e.g. nvptx)
  virtual void EnterTargetControlLoop(SourceLocation Loc, CodeGenFunction &CGF,
                                      StringRef TgtFunName);

  // \brief Code generation for closing of sequential region. Set ups the next
  // labels for special back-ends (e.g. nvptx)
  virtual void ExitTargetControlLoop(SourceLocation Loc, CodeGenFunction &CGF,
                                     bool prevIsParallel, StringRef TgtFunName);

  // \brief Function to close an openmp region. Set the labels and generate
  // new switch case
  virtual void GenerateNextLabel(CodeGenFunction &CGF, bool prevIsParallel,
                                 bool nextIsParallel,
                                 const char *CaseBBName = 0);

  // \brief Function to open a workshare region. In NVPTX, record entering
  // workshare region in bit vector used to calculate optimal number of lanes
  // in a parallel region and update stack of pragmas
  virtual void EnterWorkshareRegion();

  // \brief Function to close a workshare region. In NVPTX, update stack of
  // pragmas
  virtual void ExitWorkshareRegion();

  virtual void GenerateIfMaster (SourceLocation Loc, CapturedStmt *CS,
      CodeGenFunction &CGF);

  // \brief Code generation when entering a simd region. For nvptx backend,
  // interact with control loop to select proper threads
  virtual void EnterSimdRegion(CodeGenFunction &CGF,
                               ArrayRef<OMPClause *> Clauses);

  // \brief Code generation when exiting a simd region. For nvptx backend,
  // interact with control loop to select proper threads in following region
  virtual void ExitSimdRegion(CodeGenFunction &CGF, llvm::Value *LoopIndex,
                              llvm::AllocaInst *LoopCount);

  // \brief Rename function if part of standard libraries. This is useful to
  // distinguish whether such functions are called from the device, as they
  // will have a different implementation than the one on the host.
  virtual StringRef RenameStandardFunction (StringRef name);

  virtual void SelectActiveThreads (CodeGenFunction &CGF);

  // these three functions are not directly called in the implementation of
  // #parallel because they are not needed for all OpenMP back-ends, but
  // only for nvptx
  virtual llvm::Value * CallParallelRegionPrepare(CodeGenFunction &CGF);

  virtual void CallParallelRegionStart(CodeGenFunction &CGF);

  virtual void CallParallelRegionEnd(CodeGenFunction &CGF);

  virtual void CallSerializedParallelStart(CodeGenFunction &CGF);

  virtual void CallSerializedParallelEnd(CodeGenFunction &CGF);

  virtual bool RequireFirstprivateSynchronization() { return true; }

  virtual void EnterParallelRegionInTarget(CodeGenFunction &CGF,
                                           OpenMPDirectiveKind DKind,
                                           ArrayRef<OpenMPDirectiveKind> SKinds,
                                           const OMPExecutableDirective &S);

  virtual void ExitParallelRegionInTarget(CodeGenFunction &CGF);

  virtual void SupportCritical (const OMPCriticalDirective &S,
      CodeGenFunction &CGF, llvm::Function * CurFn, llvm::GlobalVariable *Lck);

  virtual void EmitNativeBarrier(CodeGenFunction &CGF);

  virtual bool IsNestedParallel();

  virtual unsigned CalculateParallelNestingLevel();

  virtual llvm::Value * AllocateThreadLocalInfo(CodeGenFunction & CGF);

  virtual llvm::Value * GetNextIdIncrement(CodeGenFunction &CGF,
        bool IsStaticSchedule, const Expr * ChunkSize, llvm::Value * Chunk,
        llvm::Type * IdxTy, QualType QTy, llvm::Value * Idx,
        OpenMPDirectiveKind Kind, OpenMPDirectiveKind SKind, llvm::Value * PSt);

  // \brief Return true if the target requires a microtask for teams directive
  virtual bool requiresMicroTaskForTeams();

  // \brief Return true if the target requires a microtask for parallel
  // directive
  virtual bool requiresMicroTaskForParallel();

  // \brief Emit initialization of index for #pragma omp simd loop
  virtual void EmitSimdInitialization(llvm::Value *LoopIndex,
                                      llvm::Value *LoopCount,
                                      CodeGenFunction &CGF);

  virtual void EmitSimdIncrement(llvm::Value *LoopIndex, llvm::Value *LoopCount,
                                 CodeGenFunction &CGF);

  // to be removed
  virtual llvm::Value * Get_kmpc_print_int();
  virtual llvm::Value * Get_kmpc_print_address_int64();

  virtual llvm::Value * Get_omp_get_num_threads();
  virtual llvm::Value * Get_omp_get_num_teams();
};

/// \brief Returns an implementation of the OpenMP RT for a given target
CGOpenMPRuntime *CreateOpenMPRuntime(CodeGenModule &CGM);

} // namespace CodeGen
} // namespace clang

#endif
