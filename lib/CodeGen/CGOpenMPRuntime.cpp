//===----- CGOpenMPRuntime.cpp - Interface to OpenMP Runtimes -------------===//
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

#include "CGOpenMPRuntimeTypes.h"
#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/BitVector.h"

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "llvm/IR/Type.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace clang;
using namespace CodeGen;

// Register global initializer for OpenMP Target offloading
void CGOpenMPRuntime::registerTargetGlobalInitializer(const llvm::Constant *D){
  TargetGlobalInitializers.insert(D);
}

// Return true if D is a global initializer for OpenMP Target offloading
bool CGOpenMPRuntime::isTargetGlobalInitializer(const llvm::Constant *D){
  return TargetGlobalInitializers.count(D) != 0;
}

// Return true if the current module has global initializers
bool CGOpenMPRuntime::hasTargetGlobalInitializers(){
  return !TargetGlobalInitializers.empty();
}

// Start sharing region. This will initialize a new set of shared variables
void CGOpenMPRuntime::startSharedRegion(unsigned NestingLevel) {
  // If the current target region doesn't have any entry yet, create one. We may
  // need to create more than one entry if there were target regions with no
  // data sharing processed since the last one that does.
  if (ValuesToBeInSharedMemory.size() < NumTargetRegions)
    ValuesToBeInSharedMemory.resize(NumTargetRegions);

  auto &Levels = ValuesToBeInSharedMemory[NumTargetRegions - 1];

  // If we have no data to be shared in the nesting levels up to the current one
  // create empty arrays
  if (Levels.size() < NestingLevel + 1)
    Levels.resize(NestingLevel + 1);

  // Initiate a new set of variables for this region
  Levels.back().resize(Levels.back().size() + 1);
}

// Mark value as requiring to be moved to global memory
void CGOpenMPRuntime::addToSharedRegion(llvm::Value *V, unsigned NestingLevel) {

  // Make sure this value is not already shared.
  auto &Levels = ValuesToBeInSharedMemory[NumTargetRegions - 1];

  for (unsigned i = 0; i <= NestingLevel; ++i) {
    auto &Sets = Levels[i];
    for (auto &S : Sets)
      // Is it already shared? if so, don't add it to the sets again.
      if (S.count(V))
        return;
  }

  ValuesToBeInSharedMemory[NumTargetRegions - 1][NestingLevel].back().insert(V);
}

// Return the registered constant for a given declaration
llvm::Constant *CGOpenMPRuntime::getEntryForDeclaration(const Decl *D) {
  auto I = DeclsToEntriesMap.find(D);
  if (I != DeclsToEntriesMap.end())
    return I->getSecond();
  return nullptr;
}

// Register a function and host entry for a given directive with target
void CGOpenMPRuntime::registerEntryForDeclaration(const Decl *D,
                                                  llvm::Constant *C) {
  if (!D)
    return;
  DeclsToEntriesMap[D] = C;
}

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM)
    : CGM(CGM), DefaultOpenMPPSource(nullptr), NumTargetRegions(0),
      NumTargetGlobals(0), HasTargetInfoLoaded(false),
      TargetRegionsDescriptor(nullptr) {
  IdentTy = llvm::StructType::create(
      "ident_t", CGM.Int32Ty /* reserved_1 */, CGM.Int32Ty /* flags */,
      CGM.Int32Ty /* reserved_2 */, CGM.Int32Ty /* reserved_3 */,
      CGM.Int8PtrTy /* psource */, nullptr);
  // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
  llvm::Type *MicroParams[] = {llvm::PointerType::getUnqual(CGM.Int32Ty),
                               llvm::PointerType::getUnqual(CGM.Int32Ty)};
  Kmpc_MicroTy = llvm::FunctionType::get(CGM.VoidTy, MicroParams, true);

  // If we are in target mode, load the metadata from the host, this code has
  // to match the PostProcessModule metadata generation.

  if (!CGM.getLangOpts().OpenMPTargetMode)
    return;

  if (CGM.getLangOpts().OMPHostOutputFile.empty())
    return;

  auto Buf = llvm::MemoryBuffer::getFile(CGM.getLangOpts().OMPHostOutputFile);
  if (Buf.getError())
    return;

  llvm::LLVMContext C;
  auto ME = llvm::parseBitcodeFile(Buf.get()->getMemBufferRef(), C);

  if (ME.getError())
    return;

  llvm::NamedMDNode *MD = ME.get()->getNamedMetadata("openmp.offloading.info");
  if (!MD)
    return;

  unsigned TotalEntriesNum = 0;

  for (auto I : MD->operands()) {
    llvm::MDNode *MN = cast<llvm::MDNode>(I);
    unsigned Idx = 0;

    auto getVal = [&]() {
      llvm::ConstantAsMetadata *V =
          cast<llvm::ConstantAsMetadata>(MN->getOperand(Idx++));
      return cast<llvm::ConstantInt>(V->getValue())->getZExtValue();
    };
    auto getName = [&]() {
      llvm::MDString *V = cast<llvm::MDString>(MN->getOperand(Idx++));
      return V->getString();
    };

    switch (getVal()) {
    default:
      llvm_unreachable("Unexpected metadata!");
      break;
    case OMPTGT_METADATA_TY_GLOBAL_VAR: {
      auto &GO = GlobalsOrder[getName()];
      GO = getVal();
      ++TotalEntriesNum;
    } break;
    case OMPTGT_METADATA_TY_TARGET_REGION: {
      auto &TRO = TargetRegionsOrder[getName()];
      while (Idx < MN->getNumOperands()) {
        TRO.push_back(getVal());
        ++TotalEntriesNum;
      }
    } break;
    case OMPTGT_METADATA_TY_CTOR:
      while (Idx < MN->getNumOperands()) {
        CtorRegionsOrder.push_back(getVal());
        ++TotalEntriesNum;
      }
      break;
    case OMPTGT_METADATA_TY_DTOR: {
      auto &DRO = DtorRegionsOrder[getName()];
      DRO = getVal();
      ++TotalEntriesNum;
    } break;
    case OMPTGT_METADATA_TY_OTHER_GLOBAL_VAR:
      OtherGlobalVariables.insert(getName());
      break;
    case OMPTGT_METADATA_TY_OTHER_FUNCTION:
      OtherFunctions.insert(getName());
      break;
    }
  }

  HasTargetInfoLoaded = MD->getNumOperands();
}

CGOpenMPRuntime::~CGOpenMPRuntime() {
#ifndef NDEBUG
  if (CGM.getLangOpts().OpenMPTargetMode) {
    // Verify that all the target entries specified by the host were generated
    // by checking if the order was invalidated
    for (auto &O : GlobalsOrder)
      if (O.getValue() != -1u)
        llvm_unreachable(
            "Target global var entry was not invalidated/generated!");
    for (auto &OO : TargetRegionsOrder)
      for (auto O : OO.getValue())
        if (O != -1u)
          llvm_unreachable(
              "Target region entry was not invalidated/generated!");
    for (auto O : CtorRegionsOrder)
      if (O != -1u)
        llvm_unreachable("Target ctor was not invalidated/generated!");
    for (auto &O : DtorRegionsOrder)
      if (O.getValue() != -1u)
        llvm_unreachable("Target dtor was not invalidated/generated!");
  }
#endif
}

llvm::Value *
CGOpenMPRuntime::GetOrCreateDefaultOpenMPLocation(OpenMPLocationFlags Flags) {
  llvm::Value *Entry = OpenMPDefaultLocMap.lookup(Flags);
  if (!Entry) {
    if (!DefaultOpenMPPSource) {
      // Initialize default location for psource field of ident_t structure of
      // all ident_t objects. Format is ";file;function;line;column;;".
      // Taken from
      // http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp_str.c
      DefaultOpenMPPSource =
          CGM.GetAddrOfConstantCString(";unknown;unknown;0;0;;");
      DefaultOpenMPPSource =
          llvm::ConstantExpr::getBitCast(DefaultOpenMPPSource, CGM.Int8PtrTy);
    }
    llvm::GlobalVariable *DefaultOpenMPLocation = cast<llvm::GlobalVariable>(
        CGM.CreateRuntimeVariable(IdentTy, ".kmpc_default_loc.addr"));
    DefaultOpenMPLocation->setUnnamedAddr(true);
    DefaultOpenMPLocation->setConstant(true);
    DefaultOpenMPLocation->setLinkage(llvm::GlobalValue::PrivateLinkage);

    llvm::Constant *Zero = llvm::ConstantInt::get(CGM.Int32Ty, 0, true);
    llvm::Constant *Values[] = {Zero,
                                llvm::ConstantInt::get(CGM.Int32Ty, Flags),
                                Zero, Zero, DefaultOpenMPPSource};
    llvm::Constant *Init = llvm::ConstantStruct::get(IdentTy, Values);
    DefaultOpenMPLocation->setInitializer(Init);
    return DefaultOpenMPLocation;
  }
  return Entry;
}

llvm::Value *CGOpenMPRuntime::EmitOpenMPUpdateLocation(
    CodeGenFunction &CGF, SourceLocation Loc, OpenMPLocationFlags Flags) {
  // If no debug info is generated - return global default location.
  if (CGM.getCodeGenOpts().getDebugInfo() == CodeGenOptions::NoDebugInfo ||
      Loc.isInvalid())
    return GetOrCreateDefaultOpenMPLocation(Flags);

  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  llvm::Value *LocValue = nullptr;
  OpenMPLocMapTy::iterator I = OpenMPLocMap.find(CGF.CurFn);
  if (I != OpenMPLocMap.end()) {
    LocValue = I->second;
  } else {
    // Generate "ident_t .kmpc_loc.addr;"
    llvm::AllocaInst *AI = CGF.CreateTempAlloca(IdentTy, ".kmpc_loc.addr");
    AI->setAlignment(CGM.getDataLayout().getPrefTypeAlignment(IdentTy));
    OpenMPLocMap[CGF.CurFn] = AI;
    LocValue = AI;

    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    CGF.Builder.CreateMemCpy(LocValue, GetOrCreateDefaultOpenMPLocation(Flags),
                             llvm::ConstantExpr::getSizeOf(IdentTy),
                             CGM.PointerAlignInBytes);
  }

  // char **psource = &.kmpc_loc_<flags>.addr.psource;
  auto *PSource = CGF.Builder.CreateConstInBoundsGEP2_32(IdentTy, LocValue, 0,
                                                         IdentField_PSource);

  auto OMPDebugLoc = OpenMPDebugLocMap.lookup(Loc.getRawEncoding());
  if (OMPDebugLoc == nullptr) {
    SmallString<128> Buffer2;
    llvm::raw_svector_ostream OS2(Buffer2);
    // Build debug location
    PresumedLoc PLoc = CGF.getContext().getSourceManager().getPresumedLoc(Loc);
    OS2 << ";" << PLoc.getFilename() << ";";
    if (const FunctionDecl *FD =
            dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl)) {
      OS2 << FD->getQualifiedNameAsString();
    }
    OS2 << ";" << PLoc.getLine() << ";" << PLoc.getColumn() << ";;";
    OMPDebugLoc = CGF.Builder.CreateGlobalStringPtr(OS2.str());
    OpenMPDebugLocMap[Loc.getRawEncoding()] = OMPDebugLoc;
  }
  // *psource = ";<File>;<Function>;<Line>;<Column>;;";
  CGF.Builder.CreateStore(OMPDebugLoc, PSource);

  return LocValue;
}

llvm::Value *CGOpenMPRuntime::GetOpenMPGlobalThreadNum(CodeGenFunction &CGF,
                                                       SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  llvm::Value *GTid = nullptr;
  OpenMPGtidMapTy::iterator I = OpenMPGtidMap.find(CGF.CurFn);
  if (I != OpenMPGtidMap.end()) {
    GTid = I->second;
  } else {
    // Generate "int32 .kmpc_global_thread_num.addr;"
    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc)};
    GTid = CGF.EmitRuntimeCall(
        CreateRuntimeFunction(OMPRTL__kmpc_global_thread_num), Args);
    OpenMPGtidMap[CGF.CurFn] = GTid;
  }
  return GTid;
}

void CGOpenMPRuntime::FunctionFinished(CodeGenFunction &CGF) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  if (OpenMPGtidMap.count(CGF.CurFn))
    OpenMPGtidMap.erase(CGF.CurFn);
  if (OpenMPLocMap.count(CGF.CurFn))
    OpenMPLocMap.erase(CGF.CurFn);
}

llvm::Type *CGOpenMPRuntime::getIdentTyPointerTy() {
  return llvm::PointerType::getUnqual(IdentTy);
}

llvm::Type *CGOpenMPRuntime::getKmpc_MicroPointerTy() {
  return llvm::PointerType::getUnqual(Kmpc_MicroTy);
}

llvm::Constant *
CGOpenMPRuntime::CreateRuntimeFunction(OpenMPRTLFunction Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (Function) {
  case OMPRTL__kmpc_fork_call: {
    // Build void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro
    // microtask, ...);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                getKmpc_MicroPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_fork_call");
    break;
  }
  case OMPRTL__kmpc_global_thread_num: {
    // Build kmp_int32 __kmpc_global_thread_num(ident_t *loc);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_global_thread_num");
    break;
  }
  }
  return RTLFn;
}

#define OPENMPRTL_FUNC(name) Get_##name()
#define OPENMPRTL_ATOMIC_FUNC(QTy, Op) GetAtomicFunc(CGF, QTy, Op)
#define OPENMPRTL_ATOMIC_FUNC_GENERAL(QTyRes, QTyIn, Aop, Capture, Reverse)    \
  GetAtomicFuncGeneral(CGF, QTyRes, QTyIn, Aop, Capture, Reverse)

#define DEFAULT_EMIT_OPENMP_FUNC(name) \
  llvm::Constant* CGOpenMPRuntime::Get_##name(){                                    \
    return CGM.CreateRuntimeFunction(                                               \
            llvm::TypeBuilder<__kmpc_##name, false>::get(CGM.getLLVMContext()),     \
            "__kmpc_" #name);                                                       \
  }

#define DEFAULT_EMIT_OPENMP_FUNC_TARGET(name)                                         \
  llvm::Constant* CGOpenMPRuntime::Get_##name(){                               \
    return CGM.CreateRuntimeFunction(                                          \
            llvm::TypeBuilder<__tgt_##name, false>::get(CGM.getLLVMContext()),\
            "__tgt_" #name);                                                  \
  }

///===---------------
///
/// Default OpenMP Runtime Implementation
///
///===---------------

DEFAULT_EMIT_OPENMP_FUNC(fork_call)
DEFAULT_EMIT_OPENMP_FUNC(push_num_threads)
DEFAULT_EMIT_OPENMP_FUNC(push_proc_bind)
DEFAULT_EMIT_OPENMP_FUNC(fork_teams)
DEFAULT_EMIT_OPENMP_FUNC(push_num_teams)
DEFAULT_EMIT_OPENMP_FUNC(cancel_barrier)
DEFAULT_EMIT_OPENMP_FUNC(barrier)
DEFAULT_EMIT_OPENMP_FUNC(cancellationpoint)
DEFAULT_EMIT_OPENMP_FUNC(cancel)
DEFAULT_EMIT_OPENMP_FUNC(omp_taskyield)
DEFAULT_EMIT_OPENMP_FUNC(omp_taskwait)
DEFAULT_EMIT_OPENMP_FUNC(flush)
DEFAULT_EMIT_OPENMP_FUNC(master)
DEFAULT_EMIT_OPENMP_FUNC(end_master)
DEFAULT_EMIT_OPENMP_FUNC(single)
DEFAULT_EMIT_OPENMP_FUNC(end_single)
DEFAULT_EMIT_OPENMP_FUNC(critical)
DEFAULT_EMIT_OPENMP_FUNC(end_critical)
DEFAULT_EMIT_OPENMP_FUNC(ordered)
DEFAULT_EMIT_OPENMP_FUNC(end_ordered)
DEFAULT_EMIT_OPENMP_FUNC(end_reduce_nowait)
DEFAULT_EMIT_OPENMP_FUNC(end_reduce)
DEFAULT_EMIT_OPENMP_FUNC(atomic_start)
DEFAULT_EMIT_OPENMP_FUNC(atomic_end)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_init_4)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_init_4u)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_init_8)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_init_8u)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_next_4)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_next_4u)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_next_8)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_next_8u)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_fini_4)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_fini_4u)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_fini_8)
DEFAULT_EMIT_OPENMP_FUNC(dispatch_fini_8u)
DEFAULT_EMIT_OPENMP_FUNC(for_static_init_4)
DEFAULT_EMIT_OPENMP_FUNC(for_static_init_4u)
DEFAULT_EMIT_OPENMP_FUNC(for_static_init_8)
DEFAULT_EMIT_OPENMP_FUNC(for_static_init_8u)
DEFAULT_EMIT_OPENMP_FUNC(for_static_fini)
DEFAULT_EMIT_OPENMP_FUNC(omp_task_begin_if0)
DEFAULT_EMIT_OPENMP_FUNC(omp_task_complete_if0)
DEFAULT_EMIT_OPENMP_FUNC(omp_task_parts)
DEFAULT_EMIT_OPENMP_FUNC(taskgroup)
DEFAULT_EMIT_OPENMP_FUNC(end_taskgroup)
DEFAULT_EMIT_OPENMP_FUNC_TARGET(register_lib)
DEFAULT_EMIT_OPENMP_FUNC_TARGET(unregister_lib)

DEFAULT_EMIT_OPENMP_FUNC(threadprivate_register)
DEFAULT_EMIT_OPENMP_FUNC(global_thread_num)

DEFAULT_EMIT_OPENMP_FUNC(kernel_init)
DEFAULT_EMIT_OPENMP_FUNC(kernel_prepare_parallel)
DEFAULT_EMIT_OPENMP_FUNC(kernel_parallel)
DEFAULT_EMIT_OPENMP_FUNC(kernel_end_parallel)

DEFAULT_EMIT_OPENMP_FUNC(serialized_parallel)
DEFAULT_EMIT_OPENMP_FUNC(end_serialized_parallel)

// Special processing for __kmpc_copyprivate
// DEFAULT_GET_OPENMP_FUNC(copyprivate)
llvm::Constant *CGOpenMPRuntime::Get_copyprivate() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), CGM.SizeTy, llvm::TypeBuilder<
          void *, false>::get(C),
      llvm::TypeBuilder<kmp_reduce_func, false>::get(C), llvm::TypeBuilder<
          int32_t, false>::get(C) };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_copyprivate");
}
// Special processing for __kmpc_reduce_nowait
// DEFAULT_GET_OPENMP_FUNC(reduce_nowait)
llvm::Constant *CGOpenMPRuntime::Get_reduce_nowait() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<int32_t,
          false>::get(C), CGM.SizeTy, llvm::TypeBuilder<void *, false>::get(C),
      llvm::TypeBuilder<kmp_copy_func, false>::get(C), llvm::TypeBuilder<
          kmp_critical_name *, false>::get(C) };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_reduce_nowait");
}
// Special processing for __kmpc_reduce
// DEFAULT_GET_OPENMP_FUNC(reduce)
llvm::Constant *CGOpenMPRuntime::Get_reduce() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<int32_t,
          false>::get(C), CGM.SizeTy, llvm::TypeBuilder<void *, false>::get(C),
      llvm::TypeBuilder<kmp_copy_func, false>::get(C), llvm::TypeBuilder<
          kmp_critical_name *, false>::get(C) };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_reduce");
}
// Special processing for __kmpc_omp_task_alloc
// DEFAULT_GET_OPENMP_FUNC(omp_task_alloc)
llvm::Constant *CGOpenMPRuntime::Get_omp_task_alloc() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<int32_t,
          false>::get(C), CGM.SizeTy, CGM.SizeTy, llvm::TypeBuilder<
          kmp_routine_entry_t, false>::get(C) };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<kmp_task_t *, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_omp_task_alloc");
}
llvm::Type *CGOpenMPRuntime::getKMPDependInfoType() {
  llvm::Type *Ty = CGM.OpenMPSupport.getKMPDependInfoType();
  if (Ty)
    return Ty;
  IdentifierInfo *II = &CGM.getContext().Idents.get("__kmp_depend_info_t");
  DeclContext *DC = CGM.getContext().getTranslationUnitDecl();
  RecordDecl *RD = RecordDecl::Create(CGM.getContext(), TTK_Struct, DC,
      SourceLocation(), SourceLocation(), II);
  RD->startDefinition();
  DC->addHiddenDecl(RD);
  II = &CGM.getContext().Idents.get("base_addr");
  FieldDecl *FD = FieldDecl::Create(CGM.getContext(), RD, SourceLocation(),
      SourceLocation(), II, CGM.getContext().getIntPtrType(),
      CGM.getContext().getTrivialTypeSourceInfo(
          CGM.getContext().getIntPtrType(), SourceLocation()), 0, false,
      ICIS_NoInit);
  FD->setAccess(AS_public);
  RD->addDecl(FD);
  II = &CGM.getContext().Idents.get("len");
  FD = FieldDecl::Create(CGM.getContext(), RD, SourceLocation(),
      SourceLocation(), II, CGM.getContext().getSizeType(),
      CGM.getContext().getTrivialTypeSourceInfo(CGM.getContext().getSizeType(),
          SourceLocation()), 0, false, ICIS_NoInit);
  FD->setAccess(AS_public);
  RD->addDecl(FD);
  II = &CGM.getContext().Idents.get("flags");
  FD = FieldDecl::Create(CGM.getContext(), RD, SourceLocation(),
      SourceLocation(), II, CGM.getContext().BoolTy,
      CGM.getContext().getTrivialTypeSourceInfo(CGM.getContext().BoolTy,
          SourceLocation()), 0, false, ICIS_NoInit);
  FD->setAccess(AS_public);
  RD->addDecl(FD);
  RD->completeDefinition();
  QualType QTy = CGM.getContext().getRecordType(RD);
  Ty = CGM.getTypes().ConvertTypeForMem(QTy);
  CGM.OpenMPSupport.setKMPDependInfoType(Ty,
      CGM.getContext().getTypeAlignInChars(QTy).getQuantity());
  return Ty;
}
// Special processing for __kmpc_omp_task_with_deps
// DEFAULT_GET_OPENMP_FUNC(omp_task_with_deps)
llvm::Constant *CGOpenMPRuntime::Get_omp_task_with_deps() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<kmp_task_t *,
          false>::get(C), llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo(),
      llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_omp_task_with_deps");
}
// Special processing for __kmpc_omp_wait_deps
// DEFAULT_GET_OPENMP_FUNC(omp_wait_deps)
llvm::Constant *CGOpenMPRuntime::Get_omp_wait_deps() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<int32_t,
          false>::get(C), getKMPDependInfoType()->getPointerTo(),
      llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_omp_wait_deps");
}
// Special processing for __tgt_target
llvm::Constant *CGOpenMPRuntime::Get_target() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = {llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void *, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<int64_t *, false>::get(C),
                          llvm::TypeBuilder<int32_t *, false>::get(C)};

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target");
}
llvm::Constant *CGOpenMPRuntime::Get_target_nowait() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<int32_t, false>::get(C),
      llvm::TypeBuilder<void *, false>::get(C), llvm::TypeBuilder<
      int32_t, false>::get(C), llvm::TypeBuilder<void **, false>::get(C),
      llvm::TypeBuilder<void **, false>::get(C), llvm::TypeBuilder<int64_t *,
      false>::get(C), llvm::TypeBuilder<int32_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo(), llvm::TypeBuilder<int32_t,
      false>::get(C), getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_nowait");
}
// Special processing for __tgt_target_teams
llvm::Constant *CGOpenMPRuntime::Get_target_teams() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = {llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void *, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<int64_t *, false>::get(C),
                          llvm::TypeBuilder<int32_t *, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C)};

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_teams");
}
// Special processing for __tgt_target_teams_nowait
llvm::Constant *CGOpenMPRuntime::Get_target_teams_nowait() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<int32_t, false>::get(C),
      llvm::TypeBuilder<void *, false>::get(C), llvm::TypeBuilder<
      int32_t, false>::get(C), llvm::TypeBuilder<void **, false>::get(C),
      llvm::TypeBuilder<void **, false>::get(C), llvm::TypeBuilder<int64_t *,
      false>::get(C), llvm::TypeBuilder<int32_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<int32_t,
      false>::get(C), llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo(), llvm::TypeBuilder<int32_t,
      false>::get(C), getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<int32_t, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_teams_nowait");
}
// Special processing for __tgt_target_data_begin
llvm::Constant *CGOpenMPRuntime::Get_target_data_begin() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = {llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<int64_t *, false>::get(C),
                          llvm::TypeBuilder<int32_t *, false>::get(C)};

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_data_begin");
}
// Special processing for __tgt_target_data_begin
llvm::Constant *CGOpenMPRuntime::Get_target_data_begin_nowait() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<int32_t, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<
      void **, false>::get(C), llvm::TypeBuilder<void **, false>::get(C),
      llvm::TypeBuilder<int64_t *, false>::get(C), llvm::TypeBuilder<int32_t *,
      false>::get(C), llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo(), llvm::TypeBuilder<int32_t,
      false>::get(C), getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_data_begin_nowait");
}
// Special processing for __tgt_target_data_end
llvm::Constant *CGOpenMPRuntime::Get_target_data_end() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = {llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<int64_t *, false>::get(C),
                          llvm::TypeBuilder<int32_t *, false>::get(C)};

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_data_end");
}
// Special processing for __tgt_target_data_end
llvm::Constant *CGOpenMPRuntime::Get_target_data_end_nowait() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<int32_t, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<
      void **, false>::get(C), llvm::TypeBuilder<void **, false>::get(C),
      llvm::TypeBuilder<int64_t *, false>::get(C), llvm::TypeBuilder<int32_t *,
      false>::get(C), llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo(), llvm::TypeBuilder<int32_t,
      false>::get(C), getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_data_end_nowait");
}
// Special processing for __tgt_target_data_update
llvm::Constant *CGOpenMPRuntime::Get_target_data_update() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = {llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<int32_t, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<void **, false>::get(C),
                          llvm::TypeBuilder<int64_t *, false>::get(C),
                          llvm::TypeBuilder<int32_t *, false>::get(C)};

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_data_update");
}
// Special processing for __tgt_target_data_update
llvm::Constant *CGOpenMPRuntime::Get_target_data_update_nowait() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<int32_t, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C), llvm::TypeBuilder<
      void **, false>::get(C), llvm::TypeBuilder<void **, false>::get(C),
      llvm::TypeBuilder<int64_t *, false>::get(C), llvm::TypeBuilder<int32_t *,
      false>::get(C), llvm::TypeBuilder<int32_t, false>::get(C),
      getKMPDependInfoType()->getPointerTo(), llvm::TypeBuilder<int32_t,
      false>::get(C), getKMPDependInfoType()->getPointerTo() };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__tgt_target_data_update_nowait");
}

// Special processing for __kmpc_threadprivate_cached
// DEFAULT_GET_OPENMP_FUNC(threadprivate_cached)
llvm::Constant *CGOpenMPRuntime::Get_threadprivate_cached() {
  llvm::LLVMContext &C = CGM.getLLVMContext();
  llvm::Type *Params[] = { llvm::TypeBuilder<ident_t *, false>::get(C),
      llvm::TypeBuilder<int32_t, false>::get(C),
      llvm::TypeBuilder<void *, false>::get(C), CGM.SizeTy, llvm::TypeBuilder<
          void ***, false>::get(C) };

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::TypeBuilder<void *, false>::get(C), Params, false);
  return CGM.CreateRuntimeFunction(FT, "__kmpc_threadprivate_cached");
}

QualType CGOpenMPRuntime::GetAtomicType(CodeGenFunction &CGF, QualType QTy) {
  if (QTy->isComplexType())
    return QTy;
  if (!QTy->isArithmeticType())
    return QualType();
  if (QTy->isRealFloatingType())
    return QTy->getCanonicalTypeUnqualified(); // CGF.ConvertTypeForMem(QTy->getCanonicalTypeUnqualified());
  uint64_t TySize = CGF.getContext().getTypeSize(QTy);
  if (CGF.getContext().getTypeSize(CGF.getContext().CharTy) == TySize)
    return
        QTy->isUnsignedIntegerOrEnumerationType() ?
            CGF.getContext().UnsignedCharTy : CGF.getContext().SignedCharTy;
  else if (CGF.getContext().getTypeSize(CGF.getContext().ShortTy) == TySize)
    return
        QTy->isUnsignedIntegerOrEnumerationType() ?
            CGF.getContext().UnsignedShortTy : CGF.getContext().ShortTy;
  else if (CGF.getContext().getTypeSize(CGF.getContext().IntTy) == TySize)
    return
        QTy->isUnsignedIntegerOrEnumerationType() ?
            CGF.getContext().UnsignedIntTy : CGF.getContext().IntTy;
  else if (CGF.getContext().getTypeSize(CGF.getContext().LongTy) == TySize)
    return
        QTy->isUnsignedIntegerOrEnumerationType() ?
            CGF.getContext().UnsignedLongTy : CGF.getContext().LongTy;
  else if (CGF.getContext().getTypeSize(CGF.getContext().LongLongTy) == TySize)
    return
        QTy->isUnsignedIntegerOrEnumerationType() ?
            CGF.getContext().UnsignedLongLongTy : CGF.getContext().LongLongTy;
  else if (CGF.getContext().getTypeSize(CGF.getContext().Int128Ty) == TySize)
    return
        QTy->isUnsignedIntegerOrEnumerationType() ?
            CGF.getContext().UnsignedInt128Ty : CGF.getContext().Int128Ty;
  return QualType();
}

llvm::Value *CGOpenMPRuntime::GetAtomicFuncGeneral(CodeGenFunction &CGF,
    QualType QTyRes, QualType QTyIn, CGOpenMPRuntime::EAtomicOperation Aop,
    bool Capture, bool Reverse) {
  SmallString<40> Str;
  llvm::raw_svector_ostream OS(Str);

  if (QTyRes.isVolatileQualified() || QTyIn.isVolatileQualified())
    return 0;

  int64_t TySize =
      CGF.CGM.GetTargetTypeStoreSize(CGF.ConvertTypeForMem(QTyRes)).getQuantity();
  if (QTyRes->isRealFloatingType()) {
    OS << "__kmpc_atomic_float";
    if (TySize != 4 && TySize != 8 && TySize != 10 && TySize != 16)
      return 0;
  } else if (QTyRes->isComplexType()) {
    OS << "__kmpc_atomic_cmplx";
    if (TySize != 8 && TySize != 16)
      return 0;
  } else if (QTyRes->isScalarType()) {
    OS << "__kmpc_atomic_fixed";
    if (TySize != 1 && TySize != 2 && TySize != 4 && TySize != 8)
      return 0;
  } else
    return 0;
  // for complex type, the size is for real or imag part
  if (QTyRes->isComplexType()) {
    OS << TySize / 2;
  } else {
    OS << TySize;
  }
  switch (Aop) {
  case OMP_Atomic_orl:
    OS << "_orl";
    break;
  case OMP_Atomic_orb:
    OS << "_orb";
    break;
  case OMP_Atomic_andl:
    OS << "_andl";
    break;
  case OMP_Atomic_andb:
    OS << "_andb";
    break;
  case OMP_Atomic_xor:
    OS << "_xor";
    break;
  case OMP_Atomic_sub:
    OS << "_sub";
    break;
  case OMP_Atomic_add:
    OS << "_add";
    break;
  case OMP_Atomic_mul:
    OS << "_mul";
    break;
  case OMP_Atomic_div:
    if (QTyRes->hasUnsignedIntegerRepresentation() || QTyRes->isPointerType()) {
      if (!CGF.getContext().hasSameType(QTyIn, QTyRes))
        return 0;
      OS << "u";
    }
    OS << "_div";
    break;
  case OMP_Atomic_min:
    OS << "_min";
    break;
  case OMP_Atomic_max:
    OS << "_max";
    break;
  case OMP_Atomic_shl:
    OS << "_shl";
    break;
  case OMP_Atomic_shr:
    if (QTyRes->hasUnsignedIntegerRepresentation() || QTyRes->isPointerType()) {
      if (!CGF.getContext().hasSameType(QTyIn, QTyRes))
        return 0;
      OS << "u";
    }
    OS << "_shr";
    break;
  case OMP_Atomic_wr:
    OS << "_wr";
    break;
  case OMP_Atomic_rd:
    OS << "_rd";
    break;
  case OMP_Atomic_assign:
    return 0;
  case OMP_Atomic_invalid:
  default:
    llvm_unreachable("Unknown atomic operation.");
  }
  if (Capture) {
    OS << "_cpt";
    if (!CGF.getContext().hasSameType(QTyIn, QTyRes))
      return 0;
  }
  if (Reverse
      && (Aop == OMP_Atomic_sub || Aop == OMP_Atomic_div
          || Aop == OMP_Atomic_shr || Aop == OMP_Atomic_shl)) {
    OS << "_rev";
    if (!CGF.getContext().hasSameType(QTyIn, QTyRes))
      return 0;
  }
  int64_t TyInSize = CGF.CGM.GetTargetTypeStoreSize(
      CGF.ConvertTypeForMem(QTyIn)).getQuantity();
  if (!CGF.getContext().hasSameType(QTyIn, QTyRes)) {
    if (QTyRes->isScalarType() && QTyIn->isRealFloatingType() && TyInSize == 8)
      OS << "_float8";
    else
      return 0;
  }
  SmallVector<llvm::Type *, 5> Params;
  Params.push_back(
      llvm::TypeBuilder<ident_t, false>::get(CGF.CGM.getLLVMContext())->getPointerTo());
  Params.push_back(CGF.Int32Ty);
  llvm::Type *Ty = CGF.ConvertTypeForMem(GetAtomicType(CGF, QTyRes));
  Params.push_back(Ty->getPointerTo());
  if (Aop != OMP_Atomic_rd)
    Params.push_back(CGF.ConvertTypeForMem(GetAtomicType(CGF, QTyIn)));
  if (Capture) {
    Params.push_back(CGF.Int32Ty);
  }
  llvm::Type *RetTy = CGF.VoidTy;
  if (Capture || Aop == OMP_Atomic_rd)
    RetTy = Ty;
  llvm::FunctionType *FunTy = llvm::FunctionType::get(RetTy, Params, false);
  return CGF.CGM.CreateRuntimeFunction(FunTy, OS.str());
}

llvm::Value *CGOpenMPRuntime::GetAtomicFunc(CodeGenFunction &CGF, QualType QTy,
    OpenMPReductionClauseOperator Op) {

  if (QTy.isVolatileQualified())
    return 0;

  EAtomicOperation Aop = OMP_Atomic_invalid;
  switch (Op) {
  case OMPC_REDUCTION_or:
    Aop = OMP_Atomic_orl;
    break;
  case OMPC_REDUCTION_bitor:
    Aop = OMP_Atomic_orb;
    break;
  case OMPC_REDUCTION_and:
    Aop = OMP_Atomic_andl;
    break;
  case OMPC_REDUCTION_bitand:
    Aop = OMP_Atomic_andb;
    break;
  case OMPC_REDUCTION_bitxor:
    Aop = OMP_Atomic_xor;
    break;
  case OMPC_REDUCTION_sub:
    Aop = OMP_Atomic_add;
    break;
  case OMPC_REDUCTION_add:
    Aop = OMP_Atomic_add;
    break;
  case OMPC_REDUCTION_mult:
    Aop = OMP_Atomic_mul;
    break;
  case OMPC_REDUCTION_min:
    Aop = OMP_Atomic_min;
    break;
  case OMPC_REDUCTION_max:
    Aop = OMP_Atomic_max;
    break;
  case OMPC_REDUCTION_custom:
    return 0;
  case OMPC_REDUCTION_unknown:
  case NUM_OPENMP_REDUCTION_OPERATORS:
    llvm_unreachable("Unknown reduction operation.");
  }
  return GetAtomicFuncGeneral(CGF, QTy, QTy, Aop, false, false);
}

llvm::Value *
CGOpenMPRuntime::GetTeamReduFunc(CodeGenFunction &CGF, QualType QTy,
                                 OpenMPReductionClauseOperator Op) {
  return 0;
}

/// This is a hook to enable postprocessing of the module.
void CGOpenMPRuntime::PostProcessModule(CodeGenModule &CGM) {

  // Create the metadata with the OpenMP offloading information only for the
  // host
  if (!CGM.getLangOpts().OpenMPTargetMode) {
    llvm::SmallVector<llvm::MDNode *, 64> Nodes;
    llvm::Module &M = CGM.getModule();
    llvm::LLVMContext &C = M.getContext();

    llvm::NamedMDNode *MD =
        M.getOrInsertNamedMetadata("openmp.offloading.info");

    auto getVal = [&](unsigned v) {
      return llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), v));
    };
    auto getName = [&](StringRef v) { return llvm::MDString::get(C, v); };

    // Generate Metadata for global variables
    for (auto &I : GlobalsOrder) {
      llvm::SmallVector<llvm::Metadata *, 32> Ops;
      Ops.push_back(getVal(OMPTGT_METADATA_TY_GLOBAL_VAR));
      Ops.push_back(getName(I.first()));
      Ops.push_back(getVal(I.second));
      MD->addOperand(llvm::MDNode::get(C, Ops));
    }
    // Generate Metadata for target regions
    for (auto &I : TargetRegionsOrder) {
      llvm::SmallVector<llvm::Metadata *, 32> Ops;
      Ops.push_back(getVal(OMPTGT_METADATA_TY_TARGET_REGION));
      Ops.push_back(getName(I.first()));
      for (auto O : I.second)
        Ops.push_back(getVal(O));
      MD->addOperand(llvm::MDNode::get(C, Ops));
    }
    // Generate Metadata for Ctor regions
    if (!CtorRegionsOrder.empty()) {
      llvm::SmallVector<llvm::Metadata *, 32> Ops;
      Ops.push_back(getVal(OMPTGT_METADATA_TY_CTOR));
      for (auto O : CtorRegionsOrder)
        Ops.push_back(getVal(O));
      MD->addOperand(llvm::MDNode::get(C, Ops));
    }
    // Generate Metadata for Dtor regions
    for (auto &I : DtorRegionsOrder) {
      llvm::SmallVector<llvm::Metadata *, 32> Ops;
      Ops.push_back(getVal(OMPTGT_METADATA_TY_DTOR));
      Ops.push_back(getName(I.first()));
      Ops.push_back(getVal(I.second));
      MD->addOperand(llvm::MDNode::get(C, Ops));
    }
    // Generate Metadata for other global vars, if any
    for (auto &I : OtherGlobalVariables) {
      llvm::SmallVector<llvm::Metadata *, 32> Ops;
      Ops.push_back(getVal(OMPTGT_METADATA_TY_OTHER_GLOBAL_VAR));
      Ops.push_back(getName(I));
      MD->addOperand(llvm::MDNode::get(C, Ops));
    }
    // Generate Metadata for other functions, if any
    for (auto &I : OtherFunctions) {
      llvm::SmallVector<llvm::Metadata *, 32> Ops;
      Ops.push_back(getVal(OMPTGT_METADATA_TY_OTHER_FUNCTION));
      Ops.push_back(getName(I));
      MD->addOperand(llvm::MDNode::get(C, Ops));
    }
  } else if (!OrderForEntry.empty()) {
    // In target mode we want to ensure the ordering is consistent with what the
    // host specified with the metadata.
    llvm::Module::GlobalListType &Globals = CGM.getModule().getGlobalList();
    assert(!Globals.empty() && "We must have globals to be ordered!");

    llvm::SmallVector<llvm::Module::GlobalListType::iterator, 32> Entries(
        OrderForEntry.size(), Globals.end());

    for (auto I = Globals.begin(), E = Globals.end(); I != E; ++I) {
      // Check if we have order specified for this global, if so save it in the
      // Entries array
      auto Order = OrderForEntry.find(*&I);
      if (Order != OrderForEntry.end())
        Entries[Order->getSecond()] = I;
    }

    auto I = Globals.end();
    --I;

    // Move the entries one by one to the back of the globals list, observing
    // the order that was specified for them
    for (unsigned i = Entries.size(); i > 0; --i) {
      auto ToBeMoved = Entries[i - 1];
      assert(ToBeMoved != Globals.end() && "Invalid iterator to be moved!");
      Globals.splice(I, Globals, ToBeMoved);
      I = ToBeMoved;
    }
  }

  if (CGM.getLangOpts().OpenMPTargetMode
      && CGM.getLangOpts().OpenMPTargetIRDump)
  CGM.getModule().dump();
  if (!CGM.getLangOpts().OpenMPTargetMode
      && CGM.getLangOpts().OpenMPHostIRDump)
    CGM.getModule().dump();
}

void CGOpenMPRuntime::PostProcessTargetFunction(const Decl *D,
                                        llvm::Function *F,
                                        const CGFunctionInfo &FI){
  CGM.SetInternalFunctionAttributes(D, F, FI);
  PostProcessTargetFunction(F);
}
void CGOpenMPRuntime::PostProcessTargetFunction(llvm::Function *F) {
  // If we are in target mode all the target functions need to be externally
  // visible.
  if (CGM.getLangOpts().OpenMPTargetMode)
    F->setLinkage(llvm::GlobalValue::ExternalLinkage);
}

static llvm::Value *GEP(CGBuilderTy &B, llvm::Value *Base, int field) {
  return B.CreateConstInBoundsGEP2_32(Base->getType()->getPointerElementType(),
                                      Base, 0, field);
}

static void StoreField(CGBuilderTy &B, llvm::Value *Val, llvm::Value *Dst,
    int field) {
  B.CreateStore(Val, GEP(B, Dst, field));
}

llvm::Value *CGOpenMPRuntime::CreateIntelOpenMPRTLLoc(SourceLocation Loc,
    CodeGenFunction &CGF, unsigned Flags) {
  llvm::Value *Tmp;
  // ident_t tmp;
  llvm::AllocaInst *AI = 0;
  llvm::BasicBlock &EntryBB = CGF.CurFn->getEntryBlock();
  std::string VarName = ".__kmpc_ident_t." + llvm::utostr(Flags) + ".";
  std::string DefaultLoc = ".omp.default.loc.";
  std::string DefaultConstName = DefaultLoc + llvm::utostr(Flags) + ".";
  llvm::Value *DefaultString;
  if (!(DefaultString = CGM.getModule().getNamedValue(DefaultLoc))) {
    DefaultString = CGF.Builder.CreateGlobalString(";unknown;unknown;0;0;;",
        DefaultLoc);
  }
  for (llvm::BasicBlock::iterator I = EntryBB.begin(), E = EntryBB.end();
      I != E; ++I)
    if (I->getName().startswith(VarName)) {
      AI = cast<llvm::AllocaInst>(I);
      break;
    }
  if (!AI) {
    llvm::StructType *StTy = llvm::IdentTBuilder::get(CGM.getLLVMContext());
    AI = CGF.CreateTempAlloca(StTy, VarName);
    AI->setAlignment(CGM.PointerAlignInBytes);
    CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveIP();
    assert(SavedIP.isSet() && "No insertion point is set!");
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    llvm::Value *DefaultVal;
    if (!(DefaultVal = CGM.getModule().getNamedValue(DefaultConstName))) {
      llvm::Constant *Zero = CGF.Builder.getInt32(0);
      llvm::Value *Args[] = { Zero, Zero };
      llvm::Constant *Values[] = { Zero, CGF.Builder.getInt32(Flags), Zero,
          Zero, cast<llvm::Constant>(
              CGF.Builder.CreateInBoundsGEP(DefaultString, Args)) };
      llvm::Constant *Init = llvm::ConstantStruct::get(StTy,
          llvm::makeArrayRef(Values));
      llvm::GlobalVariable *ConstVar = new llvm::GlobalVariable(CGM.getModule(),
          StTy, true, llvm::GlobalValue::PrivateLinkage, Init,
          DefaultConstName);
      ConstVar->setUnnamedAddr(true);
      DefaultVal = ConstVar;
    }
    CGF.Builder.CreateMemCpy(AI, DefaultVal,
        llvm::ConstantExpr::getSizeOf(StTy), CGM.PointerAlignInBytes);
    CGF.Builder.restoreIP(SavedIP);
  }
  Tmp = AI;
  if (CGM.getCodeGenOpts().getDebugInfo() != CodeGenOptions::NoDebugInfo
      && Loc.isValid()) {
    PresumedLoc PLoc = CGM.getContext().getSourceManager().getPresumedLoc(Loc);
    std::string Res = ";";
    Res += PLoc.getFilename();
    Res += ";";
    if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(
        CGF.CurFuncDecl)) {
      Res += FD->getQualifiedNameAsString();
    }
    Res += ";";
    Res += llvm::utostr(PLoc.getLine()) + ";" + llvm::utostr(PLoc.getColumn())
        + ";;";
    // tmp.psource = ";file;func;line;col;;";
    StoreField(CGF.Builder, CGF.Builder.CreateGlobalStringPtr(Res), Tmp,
        llvm::IdentTBuilder::psource);
  } else if (CGM.getCodeGenOpts().getDebugInfo()
      != CodeGenOptions::NoDebugInfo) {
    llvm::Value *Zero = CGF.Builder.getInt32(0);
    llvm::Value *Args[] = { Zero, Zero };
    StoreField(CGF.Builder, CGF.Builder.CreateInBoundsGEP(DefaultString, Args),
        Tmp, llvm::IdentTBuilder::psource);
  }
  return Tmp;
}

llvm::Value *CGOpenMPRuntime::CreateOpenMPGlobalThreadNum(SourceLocation Loc,
    CodeGenFunction &CGF) {
  llvm::BasicBlock &EntryBB = CGF.CurFn->getEntryBlock();
  for (llvm::BasicBlock::iterator I = EntryBB.begin(), E = EntryBB.end();
      I != E; ++I)
    if (I->getName().startswith(".__kmpc_global_thread_num."))
      return CGF.Builder.CreateLoad(I, ".gtid.");
  llvm::AllocaInst *AI = CGF.CreateTempAlloca(CGM.Int32Ty,
      ".__kmpc_global_thread_num.");
  AI->setAlignment(4);
  CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveIP();
  assert(SavedIP.isSet() && "No insertion point is set!");
  CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
  llvm::Value *IdentT = CreateIntelOpenMPRTLLoc(Loc, CGF);
  llvm::Value *Res = CGF.EmitRuntimeCall(OPENMPRTL_FUNC(global_thread_num),
      llvm::makeArrayRef<llvm::Value *>(&IdentT, 1));
  CGF.Builder.CreateStore(Res, AI);
  CGF.Builder.restoreIP(SavedIP);
  return CGF.Builder.CreateLoad(AI, ".gtid.");
}

llvm::Value *CGOpenMPRuntime::CreateOpenMPThreadPrivateCached(const VarDecl *VD,
    SourceLocation Loc, CodeGenFunction &CGF, bool NoCast) {
  if (CGM.OpenMPSupport.hasThreadPrivateVar(VD)) {
    llvm::Type *VDTy = CGM.getTypes().ConvertTypeForMem(VD->getType());
    llvm::PointerType *PTy = llvm::PointerType::get(VDTy,
        CGM.getContext().getTargetAddressSpace(VD->getType()));
    CharUnits SZ = CGM.GetTargetTypeStoreSize(VDTy);
    std::string VarCache = CGM.getMangledName(VD).str() + ".cache.";

    llvm::Value *Args[] = { CreateIntelOpenMPRTLLoc(Loc, CGF),
        CreateOpenMPGlobalThreadNum(Loc, CGF), CGF.Builder.CreateBitCast(
            VD->isStaticLocal() ?
                CGM.getStaticLocalDeclAddress(VD) : CGM.GetAddrOfGlobal(VD),
            CGM.Int8PtrTy), llvm::ConstantInt::get(CGF.SizeTy,
            SZ.getQuantity()), CGM.getModule().getNamedValue(VarCache) };
    llvm::Value *Call = CGF.EmitRuntimeCall(
        OPENMPRTL_FUNC(threadprivate_cached), Args);
    if (NoCast)
      return Call;
    return CGF.Builder.CreateBitCast(Call, PTy);
  }
  return 0;
}

/// Remove dashes and other strange characters from the target triple
/// as they may cause some problems for the external symbols
static std::string LegalizeTripleString(llvm::Triple TargetTriple) {

  const std::string &TS = TargetTriple.getTriple();
  std::string S;
  llvm::raw_string_ostream OS(S);

  for (unsigned i = 0; i < TS.size(); ++i) {
    unsigned char c = (unsigned char) TS[i];

    if (c >= 'a' && c <= 'z') {
      OS << c;
      continue;
    }
    if (c >= 'A' && c <= 'Z') {
      OS << c;
      continue;
    }
    if (c >= '0' && c <= '9') {
      OS << c;
      continue;
    }
    if (c == '_' || c == '-') {
      OS << '_';
      continue;
    }

    OS << llvm::format("%02x", (unsigned) c);
  }

  return OS.str();
}

// default barrier emit
void CGOpenMPRuntime::EmitOMPBarrier(SourceLocation L, unsigned Flags,
                                     CodeGenFunction &CGF) {
  CGF.EmitOMPCallWithLocAndTidHelper(OPENMPRTL_FUNC(barrier), L, Flags);
}

// These are hooks for NVPTX backend: nothing is generated for other backends
void CGOpenMPRuntime::EnterTargetControlLoop(SourceLocation Loc,
                                             CodeGenFunction &CGF,
                                             StringRef TgtFunName) {}

void CGOpenMPRuntime::ExitTargetControlLoop(SourceLocation Loc,
                                            CodeGenFunction &CGF,
                                            bool prevIsParallel,
                                            StringRef TgtFunName) {}

void CGOpenMPRuntime::GenerateNextLabel(CodeGenFunction &CGF,
                                        bool prevIsParallel,
                                        bool nextIsParallel,
                                        const char *CaseBBName) {}

void CGOpenMPRuntime::EnterSimdRegion(CodeGenFunction &CGF,
                                      ArrayRef<OMPClause *> Clauses) {}

void CGOpenMPRuntime::ExitSimdRegion(CodeGenFunction &CGF,
                                     llvm::Value *LoopIndex,
                                     llvm::AllocaInst *LoopCount) {}

void CGOpenMPRuntime::EnterWorkshareRegion() {}

void CGOpenMPRuntime::ExitWorkshareRegion() {}

void CGOpenMPRuntime::GenerateIfMaster(SourceLocation Loc, CapturedStmt *CS,
                                       CodeGenFunction &CGF) {}

StringRef CGOpenMPRuntime::RenameStandardFunction (StringRef name) {
  return name;
}

void CGOpenMPRuntime::SelectActiveThreads(CodeGenFunction &CGF) {}

llvm::Value * CGOpenMPRuntime::CallParallelRegionPrepare(CodeGenFunction &CGF) {
  return 0;
}

void CGOpenMPRuntime::CallParallelRegionStart(CodeGenFunction &CGF) {}

void CGOpenMPRuntime::CallParallelRegionEnd(CodeGenFunction &CGF) {}

void CGOpenMPRuntime::CallSerializedParallelStart(CodeGenFunction &CGF) {}

void CGOpenMPRuntime::CallSerializedParallelEnd(CodeGenFunction &CGF) {}

void CGOpenMPRuntime::EnterParallelRegionInTarget(
    CodeGenFunction &CGF, OpenMPDirectiveKind DKind,
    ArrayRef<OpenMPDirectiveKind> SKinds, const OMPExecutableDirective &S) {}

void CGOpenMPRuntime::ExitParallelRegionInTarget(CodeGenFunction &CGF) {}

void CGOpenMPRuntime::SupportCritical (const OMPCriticalDirective &S,
    CodeGenFunction &CGF, llvm::Function * CurFn, llvm::GlobalVariable *Lck) {
}

void CGOpenMPRuntime::EmitNativeBarrier(CodeGenFunction &CGF) {
}

bool CGOpenMPRuntime::IsNestedParallel () {
  return false;
}

unsigned CGOpenMPRuntime::CalculateParallelNestingLevel() { return 0; }

void CGOpenMPRuntime::StartNewTargetRegion() {
}

void CGOpenMPRuntime::StartTeamsRegion() {}

llvm::Value *
CGOpenMPRuntime::AllocateThreadLocalInfo(CodeGenFunction & CGF) {
	return 0;
}

llvm::Value *
CGOpenMPRuntime::GetNextIdIncrement(CodeGenFunction &CGF,
      bool IsStaticSchedule, const Expr * ChunkSize, llvm::Value * Chunk,
      llvm::Type * IdxTy, QualType QTy, llvm::Value * Idx,
      OpenMPDirectiveKind Kind, OpenMPDirectiveKind SKind, llvm::Value * PSt) {

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *NextIdx;

  // when distribute contains a parallel for, each distribute iteration
  // executes "stride" instructions of the innermost for
  // also valid for #for simd, because we explicitly transform the
  // single
  // loop into two loops

  // special handling for composite pragmas:
  bool RequiresStride = false;
  if ((Kind == OMPD_distribute_parallel_for ||
       Kind == OMPD_distribute_parallel_for_simd ||
       Kind == OMPD_teams_distribute_parallel_for ||
       Kind == OMPD_teams_distribute_parallel_for_simd ||
       Kind == OMPD_target_teams_distribute_parallel_for ||
       Kind == OMPD_target_teams_distribute_parallel_for_simd) &&
      SKind == OMPD_distribute)
    RequiresStride = true;

  if (RequiresStride) {
    llvm::Value *Stride = Builder.CreateLoad(PSt);
    NextIdx = Builder.CreateAdd(Idx, Stride, ".next.idx.", false,
                                QTy->isSignedIntegerOrEnumerationType());
  } else
    NextIdx =
        Builder.CreateAdd(Idx, llvm::ConstantInt::get(IdxTy, 1), ".next.idx.",
                          false, QTy->isSignedIntegerOrEnumerationType());

  assert(NextIdx && "NextIdx variable not set");

  return NextIdx;
}

bool CGOpenMPRuntime::requiresMicroTaskForTeams(){
  return true;
}
bool CGOpenMPRuntime::requiresMicroTaskForParallel(){
  return true;
}

void CGOpenMPRuntime::EmitSimdInitialization(llvm::Value *LoopIndex,
                                             llvm::Value *LoopCount,
                                             CodeGenFunction &CGF) {
  CGBuilderTy &Builder = CGF.Builder;

  Builder.CreateStore(llvm::ConstantInt::get(LoopCount->getType(), 0),
                      LoopIndex);
}

void CGOpenMPRuntime::EmitSimdIncrement(llvm::Value *LoopIndex,
                                        llvm::Value *LoopCount,
                                        CodeGenFunction &CGF) {
  CGBuilderTy &Builder = CGF.Builder;

  llvm::Value *NewLoopIndex =
      Builder.CreateAdd(Builder.CreateLoad(LoopIndex),
                        llvm::ConstantInt::get(LoopCount->getType(), 1));
  Builder.CreateStore(NewLoopIndex, LoopIndex);
}

///===---------------
///
/// Generate target regions descriptor
///
///===---------------

/// Return a pointer to the device image begin.
///
llvm::Constant* CGOpenMPRuntime::GetDeviceImageBeginPointer(
                                                    llvm::Triple TargetTriple){
  return new llvm::GlobalVariable(
          CGM.getModule(),
          CGM.Int8Ty,
          true,
          llvm::GlobalValue::ExternalLinkage,
          0,
          Twine("__omptgt__img_start_")
          + Twine(LegalizeTripleString(TargetTriple)));
}

/// Return a pointer to the device image end.
///
llvm::Constant* CGOpenMPRuntime::GetDeviceImageEndPointer(
                                                    llvm::Triple TargetTriple){
  return new llvm::GlobalVariable(
          CGM.getModule(),
          CGM.Int8Ty,
          true,
          llvm::GlobalValue::ExternalLinkage,
          0,
          Twine("__omptgt__img_end_")
          + Twine(LegalizeTripleString(TargetTriple)));
}

/// Return a string with the mangled name of a target region for the given
/// module and target region index
///
std::string CGOpenMPRuntime::GetOffloadEntryMangledName() {
  return GetOffloadEntryMangledName(NumTargetRegions + NumTargetGlobals);
}
std::string CGOpenMPRuntime::GetOffloadEntryMangledName(unsigned ID) {
  std::string S;
  llvm::raw_string_ostream OS(S);

  assert(ID != -1u && "Invalid Id use in name mangling??");

  // append the module unique region index
  OS << "__omptgt__" << ID << '_' << CGM.getLangOpts().OMPModuleUniqueID << '_';

  return OS.str();
}

std::string
CGOpenMPRuntime::GetOffloadEntryMangledNameForGlobalVariable(StringRef Key,
                                                             bool Invalidate) {
  unsigned Order;
  return GetOffloadEntryMangledNameForGlobalVariable(Key, Order, Invalidate);
}
std::string CGOpenMPRuntime::GetOffloadEntryMangledNameForGlobalVariable(
    StringRef Key, unsigned &Order, bool Invalidate) {
  assert(CGM.getLangOpts().OpenMPTargetMode &&
         "This should only be used in target mode!");
  auto I = GlobalsOrder.find(Key);
  assert(I != GlobalsOrder.end() && "Invalid key being used!");
  Order = I->getValue();
  if (Invalidate)
    I->getValue() = -1u;
  return GetOffloadEntryMangledName(Order);
}
std::string
CGOpenMPRuntime::GetOffloadEntryMangledNameForTargetRegion(bool Invalidate) {
  unsigned Order;
  return GetOffloadEntryMangledNameForTargetRegion(Order, Invalidate);
}
std::string
CGOpenMPRuntime::GetOffloadEntryMangledNameForTargetRegion(unsigned &Order,
                                                           bool Invalidate) {
  assert(CGM.getLangOpts().OpenMPTargetMode &&
         "This should only be used in target mode!");
  auto I = TargetRegionsOrder.find(CurTargetParentFunctionName);
  for (auto &O : I->getValue())
    if (O != -1u) {
      Order = O;
      if (Invalidate)
        O = -1u;
      return GetOffloadEntryMangledName(Order);
    }
  llvm_unreachable("Invalid key for target mangled name!");
  return std::string();
}
std::string
CGOpenMPRuntime::GetOffloadEntryMangledNameForCtor(bool Invalidate) {
  unsigned Order;
  return GetOffloadEntryMangledNameForCtor(Order, Invalidate);
}
std::string
CGOpenMPRuntime::GetOffloadEntryMangledNameForCtor(unsigned &Order,
                                                   bool Invalidate) {
  assert(CGM.getLangOpts().OpenMPTargetMode &&
         "This should only be used in target mode!");
  for (auto &O : CtorRegionsOrder)
    if (O != -1u) {
      Order = O;
      if (Invalidate)
        O = -1u;
      return GetOffloadEntryMangledName(Order);
    }
  llvm_unreachable("Invalid key for target mangled name!");
  return std::string();
}
std::string
CGOpenMPRuntime::GetOffloadEntryMangledNameForDtor(StringRef Key,
                                                   bool Invalidate) {
  unsigned Order;
  return GetOffloadEntryMangledNameForDtor(Key, Order, Invalidate);
}
std::string CGOpenMPRuntime::GetOffloadEntryMangledNameForDtor(
    StringRef Key, unsigned &Order, bool Invalidate) {
  assert(CGM.getLangOpts().OpenMPTargetMode &&
         "This should only be used in target mode!");
  auto I = DtorRegionsOrder.find(Key);
  assert(I != DtorRegionsOrder.end() && "Invalid key being used!");
  Order = I->getValue();
  if (Invalidate)
    I->getValue() = -1u;
  return GetOffloadEntryMangledName(Order);
}

/// Return the target regions descriptor or a create a new
/// one if if does not exist
///
llvm::Constant* CGOpenMPRuntime::GetTargetRegionsDescriptor(){

  // If we created the target regions descriptor before, just return it
  if (TargetRegionsDescriptor)
    return TargetRegionsDescriptor;

  assert(!CGM.getLangOpts().OpenMPTargetMode
      && "Generating offload descriptor for target code??");

  llvm::LLVMContext &C = CGM.getModule().getContext();
  llvm::Module &M = CGM.getModule();

  //Get list of devices we care about
  const std::vector<llvm::Triple> &Devices = CGM.getLangOpts().OMPTargetTriples;

  assert(Devices.size()
      && "No devices specified while running in target mode??");

  //Type of target regions descriptor
  llvm::StructType *DescTy = llvm::TypeBuilder<__tgt_bin_desc, true>::get(C);
  //Type of device image
  llvm::StructType *DevTy = llvm::TypeBuilder<__tgt_device_image, true>::get(C);
  //Type of offload entry
  llvm::StructType *EntryTy = llvm::TypeBuilder<__tgt_offload_entry, true>::get(C);

  //No devices: return a null pointer
  if (Devices.empty())
    return llvm::ConstantExpr::getBitCast(
        llvm::Constant::getNullValue(llvm::Type::getInt8PtrTy(C)),
        DescTy->getPointerTo());

  //Create the external vars that will point to the begin and end of the
  //host entries section.
  //
  // FIXME: The names of these globals need to be consistent with the linker.
  // Maybe make the runtime class to return these strings

  llvm::GlobalVariable *HostEntriesBegin = new llvm::GlobalVariable(
      M, EntryTy, true, llvm::GlobalValue::ExternalLinkage, 0,
      "__omptgt__host_entries_begin");
  llvm::GlobalVariable *HostEntriesEnd = new llvm::GlobalVariable(
      M, EntryTy, true, llvm::GlobalValue::ExternalLinkage, 0,
      "__omptgt__host_entries_end");

  //Create all device images
  llvm::SmallVector<llvm::Constant*,4> DeviceImagesEntires;

  for (unsigned i=0; i<Devices.size(); ++i){
    llvm::Constant *Dev = llvm::ConstantStruct::get(DevTy,
        CGM.getOpenMPRuntime().GetDeviceImageBeginPointer(Devices[i]),
        CGM.getOpenMPRuntime().GetDeviceImageEndPointer(Devices[i]),
        HostEntriesBegin, HostEntriesEnd, nullptr);
    DeviceImagesEntires.push_back(Dev);
  }

  //Create device images global array
  llvm::ArrayType *DeviceImagesInitTy =
      llvm::ArrayType::get(DevTy,DeviceImagesEntires.size());
  llvm::Constant *DeviceImagesInit = llvm::ConstantArray::get(
      DeviceImagesInitTy,DeviceImagesEntires);

  llvm::GlobalVariable *DeviceImages = new llvm::GlobalVariable(
      M,
      DeviceImagesInitTy,
      true,
      llvm::GlobalValue::InternalLinkage,
      DeviceImagesInit,
      "__omptgt__device_images");

  //This is a Zero array to be used in the creation of the constant expressions
  llvm::Constant *Index[] = { llvm::Constant::getNullValue(CGM.Int32Ty),
                              llvm::Constant::getNullValue(CGM.Int32Ty)};

  //Create the target region descriptor:
  // - number of devices
  // - pointer to the devices array
  // - begin of host entries point
  // - end of host entries point
  llvm::Constant *TargetRegionsDescriptorInit = llvm::ConstantStruct::get(
      DescTy,
      llvm::ConstantInt::get(CGM.Int32Ty, Devices.size()),
      llvm::ConstantExpr::getGetElementPtr(DeviceImagesInitTy,DeviceImages,Index),
      HostEntriesBegin, HostEntriesEnd, nullptr);

  TargetRegionsDescriptor = new llvm::GlobalVariable(
        M,
        DescTy,
        true,
        llvm::GlobalValue::InternalLinkage,
        TargetRegionsDescriptorInit,
        "__omptgt__target_regions_descriptor");

  return TargetRegionsDescriptor;

}

/// hooks to register information that should match between host and target
void CGOpenMPRuntime::registerGlobalVariable(const Decl *D,
                                             llvm::GlobalVariable *GV) {
  if (CGM.getLangOpts().OpenMPTargetMode) {
    // If, in target mode, if we attempt to emit a global variable entry it
    // should be valid. The check is done by GetOffloadEntry...
    unsigned Order;
    std::string Name =
        GetOffloadEntryMangledNameForGlobalVariable(GV->getName(), Order, true);
    if (llvm::GlobalVariable *G = CreateHostEntryForTargetGlobal(D, GV, Name))
      OrderForEntry[G] = Order;
    return;
  }

  // We need to understand whether this declaration is valid for the target
  // by looking into the declarative context. If it not, we just return
  const DeclContext *DC = D->getDeclContext();
  while (DC && !DC->isOMPDeclareTarget())
    DC = DC->getParent();

  if (!DC)
    return;

  GlobalsOrder[GV->getName()] = NumTargetGlobals + NumTargetRegions;
  std::string Name = GetOffloadEntryMangledName();
  CreateHostEntryForTargetGlobal(D, GV, Name);
  ++NumTargetGlobals;
  return;
}
void CGOpenMPRuntime::registerTargetRegion(const Decl *D, llvm::Function *Fn,
                                           llvm::Function *ParentFunction) {
  if (CGM.getLangOpts().OpenMPTargetMode) {
    // If we don't have information about a parent function, we should get the
    unsigned Order;
    std::string Name = GetOffloadEntryMangledNameForTargetRegion(Order, true);
    if (llvm::GlobalVariable *G =
            CreateHostPtrForCurrentTargetRegion(D, Fn, Name))
      OrderForEntry[G] = Order;
    // We use this variable as an identifier to track the current target region
    // being processed.  This is used to map thread local shared variables to
    // a shared memory structure that is maintained per target region.
    ++NumTargetRegions;
    return;
  }

  assert(ParentFunction &&
         "A Parent function must be provided when not in target mode!");
  TargetRegionsOrder[ParentFunction->getName()].push_back(NumTargetGlobals +
                                                          NumTargetRegions);
  std::string Name = GetOffloadEntryMangledName();
  CreateHostPtrForCurrentTargetRegion(D, Fn, Name);
  ++NumTargetRegions;
  return;
}
void CGOpenMPRuntime::registerCtorRegion(llvm::Function *Fn) {
  if (CGM.getLangOpts().OpenMPTargetMode) {
    unsigned Order;
    std::string Name = GetOffloadEntryMangledNameForCtor(Order, true);
    if (llvm::GlobalVariable *G =
            CreateHostPtrForCurrentTargetRegion(nullptr, Fn, Name))
      OrderForEntry[G] = Order;
    return;
  }

  CtorRegionsOrder.push_back(NumTargetGlobals + NumTargetRegions);
  std::string Name = GetOffloadEntryMangledName();
  CreateHostPtrForCurrentTargetRegion(nullptr, Fn, Name);
  ++NumTargetRegions;
  return;
}
void CGOpenMPRuntime::registerDtorRegion(llvm::Function *Fn,
                                         llvm::Constant *Destructee) {
  if (CGM.getLangOpts().OpenMPTargetMode) {
    unsigned Order;
    std::string Name =
        GetOffloadEntryMangledNameForDtor(Destructee->getName(), Order, true);
    if (llvm::GlobalVariable *G =
            CreateHostPtrForCurrentTargetRegion(nullptr, Fn, Name))
      OrderForEntry[G] = Order;
    return;
  }

  DtorRegionsOrder[Destructee->getName()] = NumTargetGlobals + NumTargetRegions;
  std::string Name = GetOffloadEntryMangledName();
  CreateHostPtrForCurrentTargetRegion(nullptr, Fn, Name);
  ++NumTargetRegions;
  return;
}
void CGOpenMPRuntime::registerOtherGlobalVariable(const VarDecl *Other) {
  llvm_unreachable("We are not using this for the moment!");
}
void CGOpenMPRuntime::registerOtherFunction(const FunctionDecl *Other,
                                            StringRef Name) {
  if (CGM.getLangOpts().OpenMPTargetMode)
    llvm_unreachable("We are not using this for the moment in target mode!");

  // Register lambda functions used in target regions
  if (CGM.OpenMPSupport.getTarget())
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Other)) {
      const CXXRecordDecl *RD = MD->getParent();
      if (RD->isLambda()) {
        OtherFunctions.insert(Name);
        return;
      }
    }

  // We need to understand whether this declaration is valid for the target
  // by looking into the declarative context. If it not, we just return
  const DeclContext *DC = Other->getDeclContext();
  while (DC && !DC->isOMPDeclareTarget()) {
    DC = DC->getParent();
  }

  if (!DC)
    return;

  OtherFunctions.insert(Name);
}

// Return true if there is any OpenMP target code to be generated
bool CGOpenMPRuntime::hasAnyTargetCodeToBeEmitted() {
  return HasTargetInfoLoaded;
}

// Return true if the given name maps to any valid target global variable
// (entry point or not
bool CGOpenMPRuntime::isValidAnyTargetGlobalVariable(StringRef name) {
  return isValidEntryTargetGlobalVariable(name) ||
         isValidOtherTargetGlobalVariable(name);
}
bool CGOpenMPRuntime::isValidAnyTargetGlobalVariable(const Decl *D) {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    return isValidAnyTargetGlobalVariable(CGM.getMangledName(GlobalDecl(VD)));
  }
  return false;
}

// Return true if the given name maps to a valid target global variable that
// is also an entry point
bool CGOpenMPRuntime::isValidEntryTargetGlobalVariable(StringRef name) {
  if (GlobalsOrder.find(name) != GlobalsOrder.end())
    return true;
  return false;
}

// Return true if the given name maps to a function that contains target
// regions that should be emitted
bool CGOpenMPRuntime::isValidTargetRegionParent(StringRef name) {
  if (TargetRegionsOrder.find(name) != TargetRegionsOrder.end())
    return true;
  return false;
}

// Return true if the given name maps to a target global variable
bool CGOpenMPRuntime::isValidOtherTargetGlobalVariable(StringRef name) {
  return OtherGlobalVariables.count(name);
}

// Return true if the given name maps to a target function
bool CGOpenMPRuntime::isValidOtherTargetFunction(StringRef name) {
  return OtherFunctions.count(name);
}

/// Return host pointer for the current target regions. This creates
/// the offload entry for the target region.
///
llvm::GlobalVariable *CGOpenMPRuntime::CreateHostPtrForCurrentTargetRegion(
    const Decl *D, llvm::Function *Fn, StringRef Name) {

  llvm::LLVMContext &C = CGM.getModule().getContext();
  llvm::Module &M = CGM.getModule();

  // Create the unique host pointer for a target region. We do not use the
  // outlined function address in the host so that it can be inlined by the
  // optimizer if appropriate.
  // In the offloading scheme, the content being pointed by this pointer is not
  // relevant. Nevertheless, we fill this content with a string that
  // correspond to the entries' name. This information can be
  // useful for some targets to expedite the runtime look-up of the entries
  // in the target image. In order to use this information the target OpenMP
  // codegen class should encode the host entries in his image.
  //
  // However, for the target code we use the function pointer since it can be
  // used to more quickly load the target functions by the runtime if it can
  // rely on the order of the entries.

  llvm::Constant *FuncPtr = llvm::ConstantExpr::getBitCast(Fn, CGM.VoidPtrTy);
  llvm::Constant *StrPtrInit = llvm::ConstantDataArray::getString(C,Name);

  llvm::GlobalVariable *Str = new llvm::GlobalVariable(
      M, StrPtrInit->getType(), true, llvm::GlobalValue::InternalLinkage,
      StrPtrInit, Twine(Name) + Twine("_entry_name"));

  llvm::Constant *StrPtr = llvm::ConstantExpr::getBitCast(Str,CGM.Int8PtrTy);

  // Create the entry struct
  // - pointer
  // - name
  // - size - we assume size zero for functions

  // Type of the entry
  llvm::StructType *EntryTy = llvm::TypeBuilder<__tgt_offload_entry, true>::get(C);

  llvm::Constant *EntryInit = llvm::ConstantStruct::get(EntryTy, FuncPtr, StrPtr,
          llvm::ConstantInt::get(CGM.Int64Ty, 0), NULL);

  llvm::GlobalVariable *Entry = new llvm::GlobalVariable(
      M, EntryTy, true, llvm::GlobalValue::ExternalLinkage, EntryInit,
      Twine(Name) + Twine("_entry"));

  // The entry has to be created in the section the linker expects it to be
  Entry->setSection(".openmptgt_host_entries");
  // We can't have any padding between symbols, so we need to have 1-byte
  // alignment
  Entry->setAlignment(1);

  // Record the pair Declaration - Function
  registerEntryForDeclaration(D, Fn);

  return Entry;
}

/// Creates the host entry for a given global and places it in the entries
/// reserved section
///
llvm::GlobalVariable *CGOpenMPRuntime::CreateHostEntryForTargetGlobal(
    const Decl *D, llvm::GlobalVariable *GV, StringRef Name) {
  assert(isValidEntryTargetGlobalVariable(GV->getName()) &&
         "Must be valid entry!");

  llvm::LLVMContext &C = CGM.getModule().getContext();
  llvm::Module &M = CGM.getModule();

  // If this entry has static storage class, we mangle the name so that it is
  // safe to export that so it can loaded by the runtime libraries
  const VarDecl *VD = cast<VarDecl>(D);
  std::string SymName;
  if (VD->getStorageClass() == SC_Static) {
    SymName = "__omptgt__static_";
    SymName += CGM.getLangOpts().OMPModuleUniqueID;
    SymName += "__";
  }

  SymName += GV->getName();

  llvm::Constant *StrPtrInit = llvm::ConstantDataArray::getString(C, SymName);

  llvm::GlobalVariable *Str = new llvm::GlobalVariable(
      M, StrPtrInit->getType(), true, llvm::GlobalValue::InternalLinkage,
      StrPtrInit, Twine(Name) + Twine("_entry_name"));

  llvm::Constant *StrPtr = llvm::ConstantExpr::getBitCast(Str,CGM.Int8PtrTy);

  // Create the entry struct
  // - pointer
  // - name
  // - size - we get the size of the global based on the datalayout

  // Type of the entry
  llvm::StructType *EntryTy = llvm::TypeBuilder<__tgt_offload_entry, true>::get(C);

  llvm::Constant *EntryInit = llvm::ConstantStruct::get(EntryTy,
          llvm::ConstantExpr::getBitCast(GV,CGM.VoidPtrTy),
          StrPtr,
          llvm::ConstantInt::get(
              CGM.Int64Ty,
              CGM.getDataLayout().getTypeStoreSize(
                  GV->getType()->getPointerElementType())),
          NULL);

  llvm::GlobalVariable *Entry = new llvm::GlobalVariable(
      M, EntryTy, true, llvm::GlobalValue::ExternalLinkage, EntryInit,
      Twine(Name) + Twine("_entry"));

  // The entry has to be created in the section the linker expects it to be
  Entry->setSection(".openmptgt_host_entries");
  // We can't have any padding between symbols, so we need to have 1-byte
  // alignment
  Entry->setAlignment(1);

  // Record the new entry associated with the provided declaration
  registerEntryForDeclaration(D, GV);

  return Entry;
}

llvm::Value * CGOpenMPRuntime::Get_kmpc_print_int() {
  return 0;
}
llvm::Value * CGOpenMPRuntime::Get_kmpc_print_address_int64() {
  return 0;
}

llvm::Value * CGOpenMPRuntime::Get_omp_get_num_threads() {
    return CGM.CreateRuntimeFunction(
       llvm::TypeBuilder<omp_get_num_threads, false>::get(
           CGM.getLLVMContext()), "omp_get_num_threads");
  }

llvm::Value * CGOpenMPRuntime::Get_omp_get_num_teams() {
    return CGM.CreateRuntimeFunction(
       llvm::TypeBuilder<omp_get_num_teams, false>::get(
           CGM.getLLVMContext()), "omp_get_num_teams");
  }


///===---------------
///
/// NVPTX OpenMP Runtime Implementation
///
///===---------------

/// Target specific runtime hacks
class CGOpenMPRuntime_NVPTX: public CGOpenMPRuntime {
  StringRef ArchName;

  // Set of variables that control the stack reserved to share data across
  // threads
  enum SharedStackTy {
    // Sharing is done in global memory
    SharedStackType_Default,
    // Sharing is done in shared memory
    SharedStackType_Fast
  };

  SharedStackTy SharedStackType;
  bool SharedStackDynamicAlloc;

  // Sharing stack sizes in bytes
  uint64_t SharedStackSizePerThread[2]; // We have two sharing levels per thread
  uint64_t SharedStackSizePerTeam;
  uint64_t SharedStackSize;

  // Set of global values that are static target entries and should therefore
  // be turned visible
  llvm::SmallSet<llvm::GlobalVariable *, 32> StaticEntries;

  // this is the identifier of a master thread, either in a block, warp or
  // entire grid, for each dimension (e.g. threadIdx.x, y and z)
  unsigned MASTER_ID;

  // type of thread local info (will be stored in loc variable)
  llvm::StructType *LocalThrTy;

  // Master and others label used by the master to control execution of threads
  // in same team
  llvm::GlobalVariable *MasterLabelShared;
  llvm::GlobalVariable *OthersLabelShared;

  // region labels associated to basic blocks and id generator
  std::vector<llvm::BasicBlock *> regionLabelMap;
  unsigned NextId;

  // Starting and ending blocks for control-loop
  llvm::BasicBlock *StartControl;
  llvm::BasicBlock *EndControl;

  // finished is private to each thread and controls ends of control-loop
  llvm::AllocaInst *FinishedVar;

  // minimal needed blocks to build up a control loop
  llvm::BasicBlock *SequentialStartBlock; // first sequential block
  llvm::BasicBlock *CheckFinished;        // while(!finished) block
  llvm::BasicBlock *FinishedCase; // block in which we set finished to true
  llvm::BasicBlock *SynchronizeAndNextState; // synchronization point
  llvm::BasicBlock *EndTarget;               // return

  // only one parallel region is currently activated as parallel in nvptx,
  // the others are just serialized (use a stack)
  typedef llvm::SmallVector<bool, 16> NestedParallelStackTy;
  NestedParallelStackTy NestedParallelStack;

  enum OMPRegionTypes {
    OMP_InitialTarget,  // every stack starts with this
    OMP_TeamSequential, // if target teams, this is used on top of target
    OMP_Parallel,
    OMP_Sequential,
    OMP_Simd,
    OMP_For // Add more worksharing constructs as necessary
  };
  // hardly there will be more than 3 nested regions
  typedef llvm::SmallVector<OMPRegionTypes, 8> OMPRegionTypesStackTy;
  OMPRegionTypesStackTy OMPRegionTypesStack;

  // pragmas inside parallel influence amount of lanes and threads (non lanes)
  // that will be used for the execution of the #parallel region. The following
  // enum is used to give priorities to constructs:
  // - #for wins over #simd and #for simd and it uses all cuda threads
  // as openmp threads
  // - #for simd and #simd select number of lanes = warpSize
  // - no #for #simd or #for simd pragmas in a parallel region: number of lanes
  // = 0
  enum IterativePragmaPriority { FOR = 0, FORSIMD = 1, SIMD = 2, ELSE = 3 };

  // The following vector and pointer into it are used to determine the amount
  // of simd lanes to be used in a #parallel region
  // FIXME: once SmallBitVector implements a operator[] lvalue, switch to it
  // expected maximum number of worksharing nests in each #parallel region
  int EXPECTED_WS_NESTS = 8;
  llvm::BitVector SimdAndWorksharingNesting;
  unsigned NextBitSimdAndWorksharingNesting;

  // this will give more resources to #simd regions: toggle to false to give
  // priority to #for (worksharing) regions
  bool MaximizeSimdPolicy = true;

  // when finished generating code for a target region, this variable contains
  // the number of lanes per thread required
  uint8_t NumSimdLanesPerTargetRegion;

  // when entering a #parallel region, record here the instrction calling
  // *prepare_parallel that will be used when closing the region to set
  // the optimal number of lanes (post-analysis of #parallel region)
  llvm::Instruction *OptimalNumLanesSetPoint;

  // Varialbe that keeps the number of parallel regions nesting. It is
  // incremented each time a parallel region is entered and decremented when
  // the same region is exited.
  llvm::AllocaInst *ParallelNesting;

  // guard for the switch (switch (NextState) { case... }
  llvm::AllocaInst *NextState;

  // this is an array with two positions two prevent race conditions due
  // to non participating threads arriving too early to read next state
  llvm::GlobalVariable *ControlState;

  // index from which we will read the next case label in ControlState
  // it is either 0 or 1
  llvm::AllocaInst *ControlStateIndex;

  // number of threads that participate in parallel region multiplied by
  // number of simd lanes associated to each such thread
  llvm::GlobalVariable *CudaThreadsInParallel;

  // number of lanes to be used when we hit first #simd level
  llvm::GlobalVariable *SimdNumLanes;

  // initial value for SimdNumLanes
  int WARP_SIZE = 32; // should obtain from parameters of target function

  // Identifier of CUDA thread as a lane
  llvm::AllocaInst *SimdLaneNum;

  llvm::SwitchInst *ControlSwitch;

  // default labels
  int FinishedState = -1;
  int FirstState = 0;

  // temporary: remember if a simd construct has a reduction clause
  bool SimdHasReduction;

  llvm::GlobalVariable *ThreadLimitGlobal;

  llvm::GlobalVariable *getMasterLabelShared() const {
    return MasterLabelShared;
  }

   void setMasterLabelShared (llvm::GlobalVariable * _masterLabelShared) {
     MasterLabelShared = _masterLabelShared;
   }

   llvm::GlobalVariable * getOthersLabelShared () const {
     return OthersLabelShared;
   }

   void setOthersLabelShared (llvm::GlobalVariable * _othersLabelShared) {
     OthersLabelShared = _othersLabelShared;
   }

   // Return basic block corresponding to label
   llvm::BasicBlock * getBasicBlockByLabel (unsigned label) const {
     return regionLabelMap[label];
   }

   // Return a reference to the entire regionLabelMap
   std::vector<llvm::BasicBlock *>& getRegionLabelMap () {
     return regionLabelMap;
   }

   llvm::BasicBlock *getEndControlBlock() const { return EndControl; }

   llvm::BasicBlock *getCheckFinished() const { return CheckFinished; }

   llvm::BasicBlock * getSequentialStartBlock () const {
     return SequentialStartBlock;
   }

  llvm::Function * Get_num_teams() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
        llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  }
  llvm::Function * Get_team_num() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
        llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  }
  llvm::Function * Get_num_threads() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
        llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  }
  llvm::Function * Get_thread_num() {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  }

  llvm::Function *Get_malloc() {
    llvm::Module *M = &CGM.getModule();
    llvm::Function *F = M->getFunction("malloc");

    if (!F) {
      llvm::FunctionType *FTy =
          llvm::FunctionType::get(CGM.VoidPtrTy, CGM.SizeTy, false);
      F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                 "malloc", M);
    }
    return F;
  }

  llvm::Function * Get_syncthreads () {
    return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
          llvm::Intrinsic::nvvm_barrier0);
  }

  // generate llvm.nvvm.ptr.gen.to.local.*
  llvm::Function * Get_ConvGenericPtrToLocal (llvm::Type * convType) {
    llvm::Type * args [] = {convType, convType};
      return llvm::Intrinsic::getDeclaration(&CGM.getModule(),
            llvm::Intrinsic::nvvm_ptr_gen_to_global, makeArrayRef(args));
    }

  // TODO: replace std vector with a working map and use index!!
  int AddNewRegionLabel (llvm::BasicBlock * bb) {
    regionLabelMap.push_back(bb);
    return NextId++;
  }

  int AddNewRegionLabelAndSwitchCase (llvm::BasicBlock * bb,
      CodeGenFunction &CGF) {
    regionLabelMap.push_back(bb);

    // TODO: make sure that the CGF is set to the proper block...if it is needed
    ControlSwitch->addCase(CGF.Builder.getInt32(NextId), bb);

    return NextId++;
  }

  bool NextOnParallelStack () {
      return NestedParallelStack.back();
  }

  void PushNewParallelRegion (bool IsParallel) {
    NestedParallelStack.push_back(IsParallel);
  }

  bool PopParallelRegion () {
    bool cont = NextOnParallelStack();
    NestedParallelStack.pop_back();
    return cont;
  }

  // return true if in nested parallel, false if not in nested parallel or
  // not in parallel at all
  bool IsNestedParallel () {
    return NextOnParallelStack();
  }

  // enquiry functions for openmp stack

  // Determine if in nested parallel region
  // This means that at least two OMP_Parallel items are found
  // in the OMP stack
  bool InNestedParallel() {
    int NumParallel = 0;
    for (OMPRegionTypesStackTy::iterator it = OMPRegionTypesStack.begin();
         it != OMPRegionTypesStack.end(); it++) {
      if (*it == OMP_Parallel)
        NumParallel++;
      if (NumParallel >= 2)
        return true;
    }
    return false;
  }

  bool InParallel() {
    for (OMPRegionTypesStackTy::iterator it = OMPRegionTypesStack.begin();
         it != OMPRegionTypesStack.end(); it++) {
      if (*it == OMP_Parallel)
        return true;
    }
    return false;
  }

  int NumParallel() {
    int NumParallel = 0;
    for (OMPRegionTypesStackTy::iterator it = OMPRegionTypesStack.begin();
         it != OMPRegionTypesStack.end(); it++) {
      if (*it == OMP_Parallel)
        NumParallel++;
    }
    return NumParallel;
  }

  // return true if the stack already contains a worksharing or simd construct
  bool InWorksharing() {
    for (OMPRegionTypesStackTy::iterator it = OMPRegionTypesStack.begin();
         it != OMPRegionTypesStack.end(); it++) {
      if (*it == OMP_For || *it == OMP_Simd)
        return true;
    }
    return false;
  }

  // access functions for SimdAndWorksharingNesting
  void AddSimdPragmaToCurrentWorkshare() {
    if (!MaximizeSimdPolicy)
      SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] =
          SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] & 1;
    else
      SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] =
          SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] | 1;
  }

  void AddForPragmaToCurrentWorkshare() {
    if (!MaximizeSimdPolicy)
      SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] =
          SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] & 0;
    else
      SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] =
          SimdAndWorksharingNesting[NextBitSimdAndWorksharingNesting] | 0;
  }

  void ForwardCurrentNestingWorkshare() {
    // double size if we ran out of bits
    if (SimdAndWorksharingNesting.size() <= NextBitSimdAndWorksharingNesting)
      SimdAndWorksharingNesting.resize(SimdAndWorksharingNesting.size() * 2);
    NextBitSimdAndWorksharingNesting++;
  }

  int CalculateNumLanes() {
    // if empty (no worksharing constructs or #simd), use only one lane
    if (SimdAndWorksharingNesting.empty())
      return 1;

    // if we maximize the number of simd lanes, and there is at least a 0 set
    // it means that there is a #simd in the parallel region, then return
    // warpSize lanes; otherwise, only one lane (no #simd)
    if (MaximizeSimdPolicy && SimdAndWorksharingNesting.any())
      return WARP_SIZE;

    // if we do not maximize the number of lanes, always return 1 unless
    // there are only #simd pragmas in the #parallel region under analysis
    if (!MaximizeSimdPolicy && SimdAndWorksharingNesting.all())
      return WARP_SIZE;

    // all other cases, return 1 lane
    return 1;
  }

  void dumpSimdAndWorksharingNesting() {
    llvm::dbgs() << "Simd and Worksharing Bit Vector:\n";
    for (unsigned i = 0; i < SimdAndWorksharingNesting.size(); i++)
      llvm::dbgs() << i << SimdAndWorksharingNesting[i];
    llvm::dbgs() << "\n";
  }

  // Return the number of nested #simd and #parallel at any time in code
  // generation by analyzing the pragma stack
  unsigned CalculateParallelNestingLevel() {
    unsigned level = 0;
    for (OMPRegionTypesStackTy::iterator it = OMPRegionTypesStack.begin();
         it != OMPRegionTypesStack.end(); it++)
      if (*it == OMP_Parallel || *it == OMP_Simd)
        level++;

    return level;
  }

  uint8_t GetNumSimdLanesPerTargetRegion() {
    return NumSimdLanesPerTargetRegion;
  }

  void SetNumSimdLanesPerTargetRegion(uint8_t _GetNumSimdLanesPerTargetRegion) {
    NumSimdLanesPerTargetRegion = _GetNumSimdLanesPerTargetRegion;
  }

  // doing a barrier in NVPTX requires handling the control loop: add a new
  // region and have all threads synchronize at the single control loop barrier
  void EmitOMPBarrier(SourceLocation L, unsigned Flags, CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;
    // generate new switch case, then look at region stack and generate thread
    // exclusion code

    // TODO: make more complex!
    // Two cases:
    // 1. We are in a non nested parallel region and we hit any kind of barrier
    // like one at the end of a #for or an explicit one. In this case, exclude
    // all lanes and non participating threads.
    // 2. We are in a nested parallel or not in a parallel region at all. In
    // this
    // case, exclude all threads except the (team) master

    // case 1
    if (NumParallel() == 1) {
      // create a new label for codegen after barrier
      llvm::BasicBlock *NextRegionBlock = llvm::BasicBlock::Create(
          CGM.getLLVMContext(), "after.barrier.check.", CGF.CurFn);

      int NextLabel = AddNewRegionLabel(NextRegionBlock);
      ControlSwitch->addCase(Bld.getInt32(NextLabel), NextRegionBlock);

      // set next label by the master only
      llvm::BasicBlock *OnlyMasterSetNext = llvm::BasicBlock::Create(
          CGM.getLLVMContext(), ".master.only.next.label", CGF.CurFn);

      llvm::Value *callThreadNum = Bld.CreateCall(Get_thread_num(), {});
      llvm::Value *AmINotMaster =
          Bld.CreateICmpNE(callThreadNum, Bld.getInt32(MASTER_ID), "NotMaster");

      Bld.CreateCondBr(AmINotMaster, SynchronizeAndNextState,
                       OnlyMasterSetNext);
      Bld.SetInsertPoint(OnlyMasterSetNext);

      // set the next label
      SmallVector<llvm::Value *, 2> GEPIdxs;
      GEPIdxs.push_back(Bld.getInt32(0));
      GEPIdxs.push_back(Bld.CreateLoad(ControlStateIndex));
      llvm::Value *NextStateValPtr = Bld.CreateGEP(ControlState, GEPIdxs);
      Bld.CreateStore(Bld.getInt32(NextLabel), NextStateValPtr);

      Bld.CreateBr(SynchronizeAndNextState);

      // start inserting new region statements into next switch case
      Bld.SetInsertPoint(NextRegionBlock);

      llvm::Value *NeedToBreak =
          Bld.CreateICmpNE(Bld.CreateLoad(SimdLaneNum), Bld.getInt32(0));

      llvm::BasicBlock *AfterBarrierCodeGen = llvm::BasicBlock::Create(
          CGM.getLLVMContext(), "after.barrier.codegen.", CGF.CurFn);

      Bld.CreateCondBr(NeedToBreak, SynchronizeAndNextState,
                       AfterBarrierCodeGen);
      Bld.SetInsertPoint(AfterBarrierCodeGen);
    } else if (NumParallel() == 0 || NumParallel() > 1) {
      // case 2
    } else
      assert(true &&
             "Number of OMP parallel regions cannot be a negative number");
  }

  // For NVTPX the control loop is generated when a target construct is found
  void EnterTargetControlLoop(SourceLocation Loc, CodeGenFunction &CGF,
                              StringRef TgtFunName) {

    CGBuilderTy &Bld = CGF.Builder;

    // 32 bits should be enough to represent the number of basic
    // blocks in a target region
    llvm::IntegerType *VarTy = CGM.Int32Ty;

    // Create variable to trace the parallel nesting one is currently in
    ParallelNesting =
        Bld.CreateAlloca(Bld.getInt32Ty(), Bld.getInt32(1), "ParallelNesting");
    Bld.CreateStore(Bld.getInt32(0), ParallelNesting);

    // we start from the first state which is a sequential region (team-master
    // only)
    NextState =
        Bld.CreateAlloca(Bld.getInt32Ty(), Bld.getInt32(1), "NextState");
    Bld.CreateStore(Bld.getInt32(FirstState), NextState);

    ControlStateIndex = Bld.CreateAlloca(Bld.getInt32Ty(), Bld.getInt32(1),
                                         "ControlStateIndex");
    Bld.CreateStore(Bld.getInt32(0), ControlStateIndex);

    char ControlStateName[] = "__omptgt__ControlState";
    char CudaThreadsInParallelName[] = "__omptgt__CudaThreadsInParallel";
    char SimdNumLanesName[] = "__omptgt__SimdNumLanes";

    // Get the control loop state variables if they were already defined and
    // initialize them.
    if (!ControlState)
      ControlState = CGM.getModule().getGlobalVariable(ControlStateName);
    if (!CudaThreadsInParallel)
      CudaThreadsInParallel =
          CGM.getModule().getGlobalVariable(CudaThreadsInParallelName);
    if (!SimdNumLanes)
      SimdNumLanes = CGM.getModule().getGlobalVariable(SimdNumLanesName);

    llvm::Type *StaticArray = llvm::ArrayType::get(VarTy, 2);

    if (!ControlState)
      ControlState = new llvm::GlobalVariable(
          CGM.getModule(), StaticArray, false, llvm::GlobalValue::CommonLinkage,
          llvm::Constant::getNullValue(StaticArray), ControlStateName, 0,
          llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);

    if (!CudaThreadsInParallel)
      CudaThreadsInParallel = new llvm::GlobalVariable(
          CGM.getModule(), VarTy, false, llvm::GlobalValue::CommonLinkage,
          llvm::Constant::getNullValue(VarTy), CudaThreadsInParallelName, 0,
          llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);

    if (!SimdNumLanes)
      SimdNumLanes = new llvm::GlobalVariable(
          CGM.getModule(), VarTy, false, llvm::GlobalValue::CommonLinkage,
          llvm::Constant::getNullValue(VarTy), SimdNumLanesName, 0,
          llvm::GlobalVariable::NotThreadLocal, SHARED_ADDRESS_SPACE, false);

    Bld.CreateStore(llvm::Constant::getNullValue(StaticArray), ControlState);
    Bld.CreateStore(llvm::Constant::getNullValue(VarTy), CudaThreadsInParallel);

    // FIXME: Adding this store creates a racing condition has the compiler can
    // optimize two stores with a selection and a single store that happens
    // before the barrier.
    // Bld.CreateStore(llvm::Constant::getNullValue(VarTy), SimdNumLanes);

    // team-master sets the initial value for SimdNumLanes
    llvm::BasicBlock *MasterInit = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".master.init.", CGF.CurFn);

    llvm::BasicBlock *NonMasterInit = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".nonmaster.init.", CGF.CurFn);

    llvm::Value *IsTeamMaster1 =
        Bld.CreateICmpEQ(Bld.CreateCall(Get_thread_num(), {}),
                         Bld.getInt32(MASTER_ID), "IsTeamMaster");

    Bld.CreateCondBr(IsTeamMaster1, MasterInit, NonMasterInit);

    Bld.SetInsertPoint(MasterInit);

    // use all cuda threads as lanes - parallel regions will change this
    Bld.CreateStore(Bld.CreateCall(Get_num_threads(), {}), SimdNumLanes);
    Bld.CreateBr(NonMasterInit);

    Bld.SetInsertPoint(NonMasterInit);
    Bld.CreateCall(Get_syncthreads(), {});

    // finished boolean controlling the while: create and init to false
    FinishedVar =
        Bld.CreateAlloca(Bld.getInt1Ty(), Bld.getInt32(1), "finished");
    Bld.CreateStore(Bld.getInt1(false), FinishedVar);

    // set initial simd lane num, which could be changed later on
    // depending on safelen and num_threads clauses
    // this initial setting ensures that #simd will work without being nested in
    // #parallel
    SimdLaneNum =
        Bld.CreateAlloca(Bld.getInt32Ty(), Bld.getInt32(1), "SimdLaneNum");
    Bld.CreateStore(Bld.CreateAnd(Bld.CreateCall(Get_thread_num(), {}),
                                  Bld.CreateSub(Bld.CreateLoad(SimdNumLanes),
                                                Bld.getInt32(1))),
                    SimdLaneNum);

    // Create all baseline basic blocks that are needed for any target region
    // to implement the control loop (further added later by AST codegen)
    llvm::BasicBlock *StartControlLoop = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".start.control", CGF.CurFn);

    llvm::BasicBlock *SwitchBlock =
        llvm::BasicBlock::Create(CGM.getLLVMContext(), ".switch.", CGF.CurFn);

    EndTarget = llvm::BasicBlock::Create(CGM.getLLVMContext(), ".end.target",
                                         CGF.CurFn);

    llvm::BasicBlock *FirstSequentialCheck = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".seq.start.check", CGF.CurFn);

    SynchronizeAndNextState = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".sync.and.next.state", CGF.CurFn);

    llvm::BasicBlock *DefaultCase =
        llvm::BasicBlock::Create(CGM.getLLVMContext(), ".default", CGF.CurFn);

    FinishedCase = llvm::BasicBlock::Create(CGM.getLLVMContext(),
                                            ".finished.case.", CGF.CurFn);

    // while(!finished)
    Bld.CreateBr(StartControlLoop);
    Bld.SetInsertPoint(StartControlLoop);

    llvm::Value *AreWeFinished =
        Bld.CreateICmpEQ(Bld.CreateLoad(FinishedVar), Bld.getInt1(true));

    Bld.CreateCondBr(AreWeFinished, EndTarget, SwitchBlock);

    // switch(NextState)...
    Bld.SetInsertPoint(SwitchBlock);
    llvm::Value *SwitchNextState = Bld.CreateLoad(NextState);

    //    llvm::Value * PrintArgs [] = {SwitchNextState};
    //    Bld.CreateCall(Get_kmpc_print_int(), PrintArgs);

    ControlSwitch = Bld.CreateSwitch(SwitchNextState, DefaultCase);

    // we always start from sequential for master-only initialization
    // of omp library on nvptx
    ControlSwitch->addCase(Bld.getInt32(FirstState), FirstSequentialCheck);
    int GetNextLabel = AddNewRegionLabel(FirstSequentialCheck);
    assert(GetNextLabel == FirstState &&
           "First sequential state is not first in control switch!");

    FinishedState = AddNewRegionLabel(FinishedCase);
    ControlSwitch->addCase(Bld.getInt32(FinishedState), FinishedCase);

    // a bad label is not handled for now
    // (TODO: add error reporting routine following OMP standard)
    Bld.SetInsertPoint(DefaultCase);
    Bld.CreateBr(SynchronizeAndNextState);

    // warning: no need to set next label because we will not use it
    // in the switch as we will never get there thanks to the setting
    // of the finished variable
    Bld.SetInsertPoint(FinishedCase);
    Bld.CreateStore(Bld.getInt1(true), FinishedVar);
    Bld.CreateBr(SynchronizeAndNextState);

    // do not do that but implement while(!finished). This helps
    // the backend ptxas to easily prove convergence
    // Bld.CreateBr(EndTarget); // go to exit directly

    Bld.SetInsertPoint(SynchronizeAndNextState);
    Bld.CreateCall(Get_syncthreads(), {});

    SmallVector<llvm::Value *, 2> GEPIdxs;
    GEPIdxs.push_back(Bld.getInt32(0));
    GEPIdxs.push_back(Bld.CreateLoad(ControlStateIndex));
    llvm::Value *NextStateValPtr = Bld.CreateGEP(ControlState, GEPIdxs);
    llvm::Value *NextStateVal = Bld.CreateLoad(NextStateValPtr);

    Bld.CreateStore(NextStateVal, NextState);
    llvm::Value *NextXoredIndex =
        Bld.CreateXor(Bld.CreateLoad(ControlStateIndex), Bld.getInt32(1));
    Bld.CreateStore(NextXoredIndex, ControlStateIndex);

    Bld.CreateBr(StartControlLoop);

    // check if we are master, possibly break
    Bld.SetInsertPoint(FirstSequentialCheck);

    llvm::Value *CallThreadNum = Bld.CreateCall(Get_thread_num(), {});
    llvm::Value *AmITeamMaster =
        Bld.CreateICmpEQ(CallThreadNum, Bld.getInt32(MASTER_ID), "AmIMaster");

    llvm::BasicBlock *FirstSequentialContent = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".first.seq.", CGF.CurFn);

    Bld.CreateCondBr(AmITeamMaster, FirstSequentialContent,
                     SynchronizeAndNextState);

    // start codegening content of target pragma
    Bld.SetInsertPoint(FirstSequentialContent);

    // Add global for thread_limit that is kept updated by the CUDA offloading
    // RTL (one per kernel)
    // init to value (0) that will provoke default being used
    ThreadLimitGlobal = new llvm::GlobalVariable(
        CGF.CGM.getModule(), Bld.getInt32Ty(), false,
        llvm::GlobalValue::ExternalLinkage, Bld.getInt32(0),
        TgtFunName + Twine("_thread_limit"));

    // first thing of sequential region:
    // initialize the state of the OpenMP rt library on the GPU
    // and pass thread limit global content to initialize thread_limit_var ICV
    llvm::Value *InitArg[] = {Bld.CreateLoad(ThreadLimitGlobal)};
    CGF.EmitRuntimeCall(OPENMPRTL_FUNC(kernel_init), makeArrayRef(InitArg));
  }

  // \brief For NVTPX generate label setting when closing
  // a target region
  void ExitTargetControlLoop(SourceLocation Loc, CodeGenFunction &CGF,
                             bool prevIsParallel, StringRef TgtFunName) {
    CGBuilderTy &Bld = CGF.Builder;

    // Master selects the next labels for everyone
    // only need to exclude others if we are in a parallel region
    if (prevIsParallel) {
      llvm::Value *ThreadIdFinished = Bld.CreateCall(Get_thread_num(), {});
      llvm::Value *NonMasterNeedToBreak = Bld.CreateICmpNE(
          ThreadIdFinished, Bld.getInt32(MASTER_ID), "NeedToBreak");

      llvm::BasicBlock *SetFinished = llvm::BasicBlock::Create(
          CGM.getLLVMContext(), ".master.set.finished", CGF.CurFn);

      Bld.CreateCondBr(NonMasterNeedToBreak, SynchronizeAndNextState,
                       SetFinished);

      Bld.SetInsertPoint(SetFinished);
    } // otherwise, we already excluded non master threads

    SmallVector<llvm::Value *, 2> GEPIdxs;
    GEPIdxs.push_back(Bld.getInt32(0));
    GEPIdxs.push_back(Bld.CreateLoad(ControlStateIndex));
    llvm::Value *NextStateValPtr = Bld.CreateGEP(ControlState, GEPIdxs);
    Bld.CreateStore(Bld.getInt32(FinishedState), NextStateValPtr);

    Bld.CreateBr(SynchronizeAndNextState);

    Bld.SetInsertPoint(EndTarget);

    // After codegen of an entire target region, we can decide the number
    // of lanes to be used and thus set a global variable that communicates
    // to the RTL on the host the exact number of CUDA threads to launch
    // This is constant at runtime
    new llvm::GlobalVariable(CGF.CGM.getModule(), Bld.getInt8Ty(), true,
                             llvm::GlobalValue::ExternalLinkage,
                             Bld.getInt8(GetNumSimdLanesPerTargetRegion()),
                             TgtFunName + Twine("_simd_info"));
  }

  void GenerateNextLabel(CodeGenFunction &CGF, bool PrevIsParallel,
                         bool NextIsParallel, const char *CaseBBName) {

    // WARNING: the code generation for if-clause will emit first
    // the else branch (sequential) then the then branch (parallel).
    // This will provoke closure of the #parallel region in else on the
    // region stack
    // going from parallel to sequential corresponds to closing a
    // parallel region

    if (PrevIsParallel && NextIsParallel) {
      // going from #parallel into same #parallel: no need to handle region
      // stack in nvptx
      assert((OMPRegionTypesStack.back() == OMP_Parallel) &&
             "parallel region to parallel region switch, but not in parallel"
             "already");
    }
    CGBuilderTy &Bld = CGF.Builder;

    // create new basic block for next region, get a new label for it
    // and add it to the switch
    const std::string NextRegionName =
        (CaseBBName != 0) ? CaseBBName : NextIsParallel ? ".par.reg.pre"
                                                        : ".seq.reg.pre";
    llvm::BasicBlock *NextRegionBlock = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), NextRegionName, CGF.CurFn);

    int NextLabel = AddNewRegionLabel(NextRegionBlock);
    ControlSwitch->addCase(Bld.getInt32(NextLabel), NextRegionBlock);

    // end of region: master set next label. If end of parallel region
    // weed out non master thread
    if (PrevIsParallel && !NextIsParallel) {
      llvm::BasicBlock *OnlyMasterSetNext = llvm::BasicBlock::Create(
          CGM.getLLVMContext(), ".master.only.next.label", CGF.CurFn);

      llvm::Value *callThreadNum = Bld.CreateCall(Get_thread_num(), {});
      llvm::Value *AmINotMaster =
          Bld.CreateICmpNE(callThreadNum, Bld.getInt32(MASTER_ID), "NotMaster");

      Bld.CreateCondBr(AmINotMaster, SynchronizeAndNextState,
                       OnlyMasterSetNext);
      Bld.SetInsertPoint(OnlyMasterSetNext);
    }

    // set the next label
    SmallVector<llvm::Value *, 2> GEPIdxs;
    GEPIdxs.push_back(Bld.getInt32(0));
    GEPIdxs.push_back(Bld.CreateLoad(ControlStateIndex));
    llvm::Value *NextStateValPtr = Bld.CreateGEP(ControlState, GEPIdxs);
    Bld.CreateStore(Bld.getInt32(NextLabel), NextStateValPtr);

    Bld.CreateBr(SynchronizeAndNextState);

    // start inserting new region statements into next switch case
    Bld.SetInsertPoint(NextRegionBlock);

    // weed out non master threads if starting sequential region
    if (!NextIsParallel) {
      llvm::BasicBlock *OnlyMasterInSequential = llvm::BasicBlock::Create(

          CGM.getLLVMContext(), ".master.only.seq.region", CGF.CurFn);
       llvm::Value * callThreadNum = Bld.CreateCall(Get_thread_num(), {});
       llvm::Value *AmINotMaster = Bld.CreateICmpNE(
           callThreadNum, Bld.getInt32(MASTER_ID), "NotMaster");

       Bld.CreateCondBr(AmINotMaster, SynchronizeAndNextState,
                        OnlyMasterInSequential);

       Bld.SetInsertPoint(OnlyMasterInSequential);
     }
  }

  // TODO: integrate within GenerateNextLabel function
  void EnterSimdRegion(CodeGenFunction &CGF, ArrayRef<OMPClause *> Clauses) {

    // record that we hit a simd region both in the stack of pragmas and
    // in the bit vector used to calculate optimal number of lanes
    AddSimdPragmaToCurrentWorkshare();
    OMPRegionTypesStack.push_back(OMP_Simd);

    // reduction is not yet implemented: in case we have a reduction, bail
    // out special handling and go sequential
    if (Clauses.data() != nullptr)
      for (ArrayRef<OMPClause *>::iterator I = Clauses.begin(),
                                           E = Clauses.end();
           I != E; ++I)
        if (*I && (*I)->getClauseKind() == OMPC_reduction) {
          // remember about this until exit
          SimdHasReduction = true;
          CGOpenMPRuntime::EnterSimdRegion(CGF, Clauses);
          return;
        }

    CGBuilderTy &Bld = CGF.Builder;

    // create new basic block for next region, get a new label for it
    // and add it to the switch
    const std::string NextRegionName = ".start.simd.";
    llvm::BasicBlock *NextRegionBlock = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), NextRegionName, CGF.CurFn);

    int NextLabel = AddNewRegionLabel(NextRegionBlock);
    ControlSwitch->addCase(Bld.getInt32(NextLabel), NextRegionBlock);

    if (OMPRegionTypesStack.back() == OMP_Parallel) {
      // simd inside parallel region: weed out non master threads for
      // next label setting
      llvm::BasicBlock *OnlyMasterSetNext = llvm::BasicBlock::Create(
          CGM.getLLVMContext(), ".master.only.next.label", CGF.CurFn);

      llvm::Value *callThreadNum = Bld.CreateCall(Get_thread_num(), {});
      llvm::Value *AmINotMaster =
          Bld.CreateICmpNE(callThreadNum, Bld.getInt32(MASTER_ID), "NotMaster");

      Bld.CreateCondBr(AmINotMaster, SynchronizeAndNextState,
                       OnlyMasterSetNext);
      Bld.SetInsertPoint(OnlyMasterSetNext);
     }

     // set the next label
     SmallVector<llvm::Value *, 2> GEPIdxs;
     GEPIdxs.push_back(Bld.getInt32(0));
     GEPIdxs.push_back(Bld.CreateLoad(ControlStateIndex));
     llvm::Value *NextStateValPtr = Bld.CreateGEP(ControlState, GEPIdxs);
     Bld.CreateStore(Bld.getInt32(NextLabel), NextStateValPtr);

     Bld.CreateBr(SynchronizeAndNextState);

     // start inserting new region statements into next switch case
     Bld.SetInsertPoint(NextRegionBlock);

     // Increment the nesting level
     Bld.CreateStore(
         Bld.CreateAdd(Bld.CreateLoad(ParallelNesting), Bld.getInt32(1)),
         ParallelNesting);

     // handle safelen clause, if specified, first check if there are clauses
     if (Clauses.data() != nullptr)
       for (ArrayRef<OMPClause *>::iterator I = Clauses.begin(),
                                            E = Clauses.end();
            I != E; ++I)
       if (*I && (*I)->getClauseKind() == OMPC_safelen) {
         OMPClause *C = dyn_cast<OMPClause>(*I);
         RValue Len = CGF.EmitAnyExpr(cast<OMPSafelenClause>(C)->getSafelen(),
                                      AggValueSlot::ignored(), true);
         llvm::ConstantInt *Val =
             dyn_cast<llvm::ConstantInt>(Len.getScalarVal());
         assert(Val);
         Bld.CreateStore(Val, SimdNumLanes);
       }

     // in simd region, weed out lanes in excess
     llvm::BasicBlock *LaneNotInExcessBlock = llvm::BasicBlock::Create(
         CGM.getLLVMContext(), ".lane.not.in.excess.", CGF.CurFn);

     llvm::Value *IsLaneInExcess = Bld.CreateICmpSGT(
         Bld.CreateLoad(SimdLaneNum), Bld.CreateLoad(SimdNumLanes));
     Bld.CreateCondBr(IsLaneInExcess, SynchronizeAndNextState,
                      LaneNotInExcessBlock);

     // lanes not in excess execute simd region
     Bld.SetInsertPoint(LaneNotInExcessBlock);
  }

  // TODO: integrate within GenerateNextLabel function
  void ExitSimdRegion(CodeGenFunction &CGF, llvm::Value *LoopIndex,
                      llvm::AllocaInst *LoopCount) {

    assert((OMPRegionTypesStack.back() == OMP_Simd) &&
           "Exiting #simd"
           "region but never entered it");
    OMPRegionTypesStack.pop_back();

    // only the master sets the next label
     CGBuilderTy &Bld = CGF.Builder;

     // fallback to sequential if there is a reduction clause
     if (SimdHasReduction) {
       CGOpenMPRuntime::ExitSimdRegion(CGF, LoopIndex, LoopCount);

       // reset reduction flag for next simd region
       SimdHasReduction = false;
       return;
     }

     // Decrement the nesting level
     Bld.CreateStore(
         Bld.CreateSub(Bld.CreateLoad(ParallelNesting), Bld.getInt32(1)),
         ParallelNesting);

     // create new basic block for next region, get a new label for it
     // and add it to the switch
     const std::string NextRegionName =
         (OMPRegionTypesStack.back() == OMP_Parallel)
             ? ".after.simd.in.parallel"
             : "after.simd.in.seq.";
     llvm::BasicBlock *NextRegionBlock = llvm::BasicBlock::Create(
         CGM.getLLVMContext(), NextRegionName, CGF.CurFn);

     int NextLabel = AddNewRegionLabel(NextRegionBlock);
     ControlSwitch->addCase(Bld.getInt32(NextLabel), NextRegionBlock);

     // simd inside parallel region: weed out non master threads for
     // next label setting
     llvm::BasicBlock *OnlyMasterSetNext = llvm::BasicBlock::Create(
         CGM.getLLVMContext(), ".master.only.next.label", CGF.CurFn);

     llvm::Value *callThreadNum = Bld.CreateCall(Get_thread_num(), {});
     llvm::Value *AmINotMaster =
         Bld.CreateICmpNE(callThreadNum, Bld.getInt32(MASTER_ID), "NotMaster");

     Bld.CreateCondBr(AmINotMaster, SynchronizeAndNextState, OnlyMasterSetNext);
     Bld.SetInsertPoint(OnlyMasterSetNext);

     // set the next label
     SmallVector<llvm::Value *, 2> GEPIdxs;
     GEPIdxs.push_back(Bld.getInt32(0));
     GEPIdxs.push_back(Bld.CreateLoad(ControlStateIndex));
     llvm::Value *NextStateValPtr = Bld.CreateGEP(ControlState, GEPIdxs);
     Bld.CreateStore(Bld.getInt32(NextLabel), NextStateValPtr);

     Bld.CreateBr(SynchronizeAndNextState);

     // start inserting new region statements into next switch case
     Bld.SetInsertPoint(NextRegionBlock);

     // weed out cuda threads for the next region, depending if parallel
     // or sequential
     llvm::BasicBlock *NextRegion = nullptr;

     // we go back to parallel handling if we are closely nested into it or
     // if we are in #parallel for
     bool NestedInParallel = OMPRegionTypesStack.back() == OMP_Parallel;
     if (!NestedInParallel) {
       // check if we are in a #for nested inside a #parallel
       enum OMPRegionTypes PopRegion = OMPRegionTypesStack.back();
       OMPRegionTypesStack.pop_back();

       // if needed, add cases here as we keep track of other worksharing
       // constructs in the RegionTypes Stack
       if (PopRegion == OMP_For && OMPRegionTypesStack.back() == OMP_Parallel)
         NestedInParallel = true;
       OMPRegionTypesStack.push_back(PopRegion);
     }

     if (NestedInParallel) {
       // closely nested in parallel, weed out non openmp threads
       NextRegion = llvm::BasicBlock::Create(CGM.getLLVMContext(),
                                             ".par.reg.code", CGF.CurFn);

       Bld.CreateCondBr(
           Bld.CreateICmpNE(Bld.CreateLoad(SimdLaneNum), Bld.getInt32(0)),
           SynchronizeAndNextState, NextRegion);
     } else {
       // going back to team-master only region: exclude all threads execpt
       // master
       NextRegion = llvm::BasicBlock::Create(CGM.getLLVMContext(),
                                             ".seq.reg.code", CGF.CurFn);

       Bld.CreateCondBr(
           Bld.CreateICmpNE(Bld.CreateCall(Get_thread_num(), {}), Bld.getInt32(0)),
           SynchronizeAndNextState, NextRegion);
     }

     Bld.SetInsertPoint(NextRegion);

     // restore last iteration value into LoopCount variable because
     // the explicit SIMD increment is NumLanes-strided
     Bld.CreateStore(Bld.CreateLoad(LoopCount), LoopIndex);
  }

  // called when entering a workshare region
  // FIXME: only #for supported for now
  void EnterWorkshareRegion() {
    AddForPragmaToCurrentWorkshare();
    OMPRegionTypesStack.push_back(OMP_For);
  }

  // called when exiting a workshare region
  // FIXME: only #for supported for now
  void ExitWorkshareRegion() {
    assert((OMPRegionTypesStack.back() == OMP_For) &&
           "Exiting #for"
           "region but never entered it");
    OMPRegionTypesStack.pop_back();
  }

  void GenerateIfMaster(SourceLocation Loc, CapturedStmt *CS,
                        CodeGenFunction &CGF) {
    CGBuilderTy &Bld = CGF.Builder;

    llvm::BasicBlock *ifMasterBlock =
        llvm::BasicBlock::Create(CGM.getLLVMContext(), ".if.master", CGF.CurFn);

    llvm::BasicBlock *fallThroughMaster = llvm::BasicBlock::Create(
        CGM.getLLVMContext(), ".fall.through.master", CGF.CurFn);

    llvm::Value *callThreadNum = Bld.CreateCall(Get_thread_num(), {});
    llvm::Value *amIMasterCond =
        Bld.CreateICmpEQ(callThreadNum, Bld.getInt32(MASTER_ID), "amIMaster");

    Bld.CreateCondBr(amIMasterCond, ifMasterBlock, fallThroughMaster);

    Bld.SetInsertPoint(ifMasterBlock);

    CGF.EmitStmt(CS->getCapturedStmt());

    Bld.CreateBr(fallThroughMaster);

    Bld.SetInsertPoint(fallThroughMaster);

    Bld.CreateCall(Get_syncthreads(), {});
   }

   // \brief scan entire parallel region looking for #for directive.
   // return true when #for is found, false otherwise
   // note: #for simd is not considered a #for and #parallel for has to be
   // handled by the caller
   bool ParallelRegionHasOpenMPLoop(const Stmt *S) {

     if (!S)
       return false;

     // traverse all children: if #for is found, return true else
     // continue scanning subtree
     for (Stmt::const_child_iterator ii = S->child_begin(), ie = S->child_end();
          ii != ie; ++ii) {
       // if we found a #for in the current node or, recursively, in one of its
       // children directly return true without looking any more
       if (isa<OMPForDirective>(*ii))
         return true;
       if (ParallelRegionHasOpenMPLoop(*ii))
         return true;
     }

     // scanned the entire region and no #for was found
     return false;
   }

   // \brief scan entire parallel region looking for #for directive.
   // return true when #for is found, false otherwise
   // note: #for simd is not considered a #for and #parallel for has to be
   // handled by the caller
   bool ParallelRegionHasSimd(const Stmt *S) {

     if (!S)
       return false;

     // traverse all children: if #for simd or #simd is found, return true else
     // continue scanning subtree
     for (Stmt::const_child_iterator ii = S->child_begin(), ie = S->child_end();
          ii != ie; ++ii) {
       // if we found a #simd or #for simd in the current node or, recursively,
       // in one of its children directly return true without looking any more
       if (isa<OMPSimdDirective>(*ii) || isa<OMPForSimdDirective>(*ii))
         return true;
       if (ParallelRegionHasSimd(*ii))
         return true;
     }

     // scanned the entire region and no #for was found
     return false;
   }

   // \brief Scan an OpenMP #parallel region looking for #for, #simd, #for simd,
   // etc. and decide amount of lanes that can be dedicated to execute #simd
   // regions
   int CalculateNumberOfLanes(OpenMPDirectiveKind DKind,
                              ArrayRef<OpenMPDirectiveKind> SKinds,
                              const OMPExecutableDirective &S) {

     // #parallel for eliminates all #simd inside
     if (isa<OMPParallelForDirective>(S))
       return 1;

     // #parallel for simd uses WARP_SIZE lanes
     // TODO: handle the case in which #parallel for simd contain a further
     // #for
     if (isa<OMPParallelForSimdDirective>(S))
       return WARP_SIZE;

     // when there is an independent single #for, bail out and use 1 lane
     if (ParallelRegionHasOpenMPLoop(&S))
       return 1;

     // no single #for, search for #simd or #for simd and if found, select
     // WARP_SIZE lanes
     if (ParallelRegionHasSimd(&S))
       return WARP_SIZE;

     // finally, no #for, #for simd, or #simd: use 1 lane
     return 1;
   }

   llvm::StringMap<StringRef> stdFuncs;

   StringRef RenameStandardFunction (StringRef name) {

     // Fill up hashmap entries lazily
     if (stdFuncs.empty()) {

       // Trigonometric functions
       stdFuncs.insert(std::make_pair("cos", "__nv_cos"));
       stdFuncs.insert(std::make_pair("sin", "__nv_sin"));
       stdFuncs.insert(std::make_pair("tan", "__nv_tan"));
       stdFuncs.insert(std::make_pair("acos", "__nv_acos"));
       stdFuncs.insert(std::make_pair("asin", "__nv_asin"));
       stdFuncs.insert(std::make_pair("atan", "__nv_atan"));
       stdFuncs.insert(std::make_pair("atan2", "__nv_atan2"));

       stdFuncs.insert(std::make_pair("cosf", "__nv_cosf"));
       stdFuncs.insert(std::make_pair("sinf", "__nv_sinf"));
       stdFuncs.insert(std::make_pair("tanf", "__nv_tanf"));
       stdFuncs.insert(std::make_pair("acosf", "__nv_acosf"));
       stdFuncs.insert(std::make_pair("asinf", "__nv_asinf"));
       stdFuncs.insert(std::make_pair("atanf", "__nv_atanf"));
       stdFuncs.insert(std::make_pair("atan2f", "__nv_atan2f"));

       // Hyperbolic functions
       stdFuncs.insert(std::make_pair("cosh", "__nv_cosh"));
       stdFuncs.insert(std::make_pair("sinh", "__nv_sinh"));
       stdFuncs.insert(std::make_pair("tanh", "__nv_tanh"));
       stdFuncs.insert(std::make_pair("acosh", "__nv_acosh"));
       stdFuncs.insert(std::make_pair("asinh", "__nv_asinh"));
       stdFuncs.insert(std::make_pair("atanh", "__nv_atanh"));

       stdFuncs.insert(std::make_pair("coshf", "__nv_coshf"));
       stdFuncs.insert(std::make_pair("sinhf", "__nv_sinhf"));
       stdFuncs.insert(std::make_pair("tanhf", "__nv_tanhf"));
       stdFuncs.insert(std::make_pair("acoshf", "__nv_acoshf"));
       stdFuncs.insert(std::make_pair("asinhf", "__nv_asinhf"));
       stdFuncs.insert(std::make_pair("atanhf", "__nv_atanhf"));

       // Exponential and logarithm functions
       stdFuncs.insert(std::make_pair("exp", "__nv_exp"));
       stdFuncs.insert(std::make_pair("frexp", "__nv_frexp"));
       stdFuncs.insert(std::make_pair("ldexp", "__nv_ldexp"));
       stdFuncs.insert(std::make_pair("log", "__nv_log"));
       stdFuncs.insert(std::make_pair("log10", "__nv_log10"));
       stdFuncs.insert(std::make_pair("modf", "__nv_modf"));
       stdFuncs.insert(std::make_pair("exp2", "__nv_exp2"));
       stdFuncs.insert(std::make_pair("expm1", "__nv_expm1"));
       stdFuncs.insert(std::make_pair("ilogb", "__nv_ilogb"));
       stdFuncs.insert(std::make_pair("log1p", "__nv_log1p"));
       stdFuncs.insert(std::make_pair("log2", "__nv_log2"));
       stdFuncs.insert(std::make_pair("logb", "__nv_logb"));
       stdFuncs.insert(std::make_pair("scalbn", "__nv_scalbn"));
//     map.insert(std::make_pair((scalbln", ""));

       stdFuncs.insert(std::make_pair("expf", "__nv_exp"));
       stdFuncs.insert(std::make_pair("frexpf", "__nv_frexpf"));
       stdFuncs.insert(std::make_pair("ldexpf", "__nv_ldexpf"));
       stdFuncs.insert(std::make_pair("logf", "__nv_logf"));
       stdFuncs.insert(std::make_pair("log10f", "__nv_log10f"));
       stdFuncs.insert(std::make_pair("modff", "__nv_modff"));
       stdFuncs.insert(std::make_pair("exp2f", "__nv_exp2f"));
       stdFuncs.insert(std::make_pair("expm1f", "__nv_expm1f"));
       stdFuncs.insert(std::make_pair("ilogbf", "__nv_ilogbf"));
       stdFuncs.insert(std::make_pair("log1pf", "__nv_log1pf"));
       stdFuncs.insert(std::make_pair("log2f", "__nv_log2f"));
       stdFuncs.insert(std::make_pair("logbf", "__nv_logbf"));
       stdFuncs.insert(std::make_pair("scalbnf", "__nv_scalbnf"));
//     map.insert(std::make_pair("scalblnf", ""));

       // Power functions
       stdFuncs.insert(std::make_pair("pow", "__nv_pow"));
       stdFuncs.insert(std::make_pair("sqrt", "__nv_sqrt"));
       stdFuncs.insert(std::make_pair("cbrt", "__nv_cbrt"));
       stdFuncs.insert(std::make_pair("hypot", "__nv_hypot"));

       stdFuncs.insert(std::make_pair("powf", "__nv_powf"));
       stdFuncs.insert(std::make_pair("sqrtf", "__nv_sqrtf"));
       stdFuncs.insert(std::make_pair("cbrtf", "__nv_cbrtf"));
       stdFuncs.insert(std::make_pair("hypotf", "__nv_hypotf"));

       // Error and gamma functions
       stdFuncs.insert(std::make_pair("erf", "__nv_erf"));
       stdFuncs.insert(std::make_pair("erfc", "__nv_erfc"));
       stdFuncs.insert(std::make_pair("tgamma", "__nv_tgamma"));
       stdFuncs.insert(std::make_pair("lgamma", "__nv_lgamma"));

       stdFuncs.insert(std::make_pair("erff", "__nv_erff"));
       stdFuncs.insert(std::make_pair("erfcf", "__nv_erfcf"));
       stdFuncs.insert(std::make_pair("tgammaf", "__nv_tgammaf"));
       stdFuncs.insert(std::make_pair("lgammaf", "__nv_lgammaf"));

       // Rounding and remainder functions
       stdFuncs.insert(std::make_pair("ceil", "__nv_ceil"));
       stdFuncs.insert(std::make_pair("floor", "__nv_floor"));
       stdFuncs.insert(std::make_pair("fmod", "__nv_fmod"));
       stdFuncs.insert(std::make_pair("trunc", "__nv_trunc"));
       stdFuncs.insert(std::make_pair("round", "__nv_round"));
       stdFuncs.insert(std::make_pair("lround", "__nv_lround"));
       stdFuncs.insert(std::make_pair("llround", "__nv_llround"));
       stdFuncs.insert(std::make_pair("rint", "__nv_rint"));
       stdFuncs.insert(std::make_pair("lrint", "__nv_lrint"));
       stdFuncs.insert(std::make_pair("llrint", "__nv_llrint"));
       stdFuncs.insert(std::make_pair("nearbyint", "__nv_nearbyint"));
       stdFuncs.insert(std::make_pair("remainder", "__nv_remainder"));
       stdFuncs.insert(std::make_pair("remquo", "__nv_remquo"));

       stdFuncs.insert(std::make_pair("ceilf", "__nv_ceilf"));
       stdFuncs.insert(std::make_pair("floorf", "__nv_floorf"));
       stdFuncs.insert(std::make_pair("fmodf", "__nv_fmodf"));
       stdFuncs.insert(std::make_pair("truncf", "__nv_truncf"));
       stdFuncs.insert(std::make_pair("roundf", "__nv_roundf"));
       stdFuncs.insert(std::make_pair("lroundf", "__nv_lroundf"));
       stdFuncs.insert(std::make_pair("llroundf", "__nv_llroundf"));
       stdFuncs.insert(std::make_pair("rintf", "__nv_rintf"));
       stdFuncs.insert(std::make_pair("lrintf", "__nv_lrintf"));
       stdFuncs.insert(std::make_pair("llrintf", "__nv_llrintf"));
       stdFuncs.insert(std::make_pair("nearbyintf", "__nv_nearbyintf"));
       stdFuncs.insert(std::make_pair("remainderf", "__nv_remainderf"));
       stdFuncs.insert(std::make_pair("remquof", "__nv_remquof"));

       // Floating-point manipulation functions
       stdFuncs.insert(std::make_pair("copysign", "__nv_copysign"));
       stdFuncs.insert(std::make_pair("nan", "__nv_nan"));
       stdFuncs.insert(std::make_pair("nextafter", "__nv_nextafter"));
//     map.insert(std::make_pair("nexttoward", ""));

       stdFuncs.insert(std::make_pair("copysignf", "__nv_copysignf"));
       stdFuncs.insert(std::make_pair("nanf", "__nv_nanf"));
       stdFuncs.insert(std::make_pair("nextafterf", "__nv_nextafterf"));
//     map.insert(std::make_pair("nexttowardf", ""));

       // Minimum, maximu,, difference functions
       stdFuncs.insert(std::make_pair("fdim", "__nv_fdim"));
       stdFuncs.insert(std::make_pair("fmax", "__nv_fmax"));
       stdFuncs.insert(std::make_pair("fmin", "__nv_fmin"));

       stdFuncs.insert(std::make_pair("fdimf", "__nv_fdimf"));
       stdFuncs.insert(std::make_pair("fmaxf", "__nv_fmaxf"));
       stdFuncs.insert(std::make_pair("fminf", "__nv_fminf"));

       // Other functions
       stdFuncs.insert(std::make_pair("fabs", "__nv_fabs"));
       stdFuncs.insert(std::make_pair("abs", "__nv_abs"));
       stdFuncs.insert(std::make_pair("fma", "__nv_fma"));

       stdFuncs.insert(std::make_pair("fabsf", "__nv_fabsf"));
       stdFuncs.insert(std::make_pair("absf", "__nv_absf"));
       stdFuncs.insert(std::make_pair("fmaf", "__nv_fmaf"));
     }

     // If callee is standard function, change its name
     StringRef match =  stdFuncs.lookup(name);
     if (!match.empty()) {
       return match;
     }

     return name;
   }

   void SelectActiveThreads (CodeGenFunction &CGF) {

     // this is only done when in non nested parallel region
     // because in a nested parallel region there is a single thread and
     // we don't need to check
     bool CurrentIsNested = PopParallelRegion();

     // if we are in the first level, the previous position is set to false
     if (!NestedParallelStack.back()) {
     CGBuilderTy &Bld = CGF.Builder;

       // call omp_get_num_threads
     llvm::Value * NumThreads = Bld.CreateCall(Get_omp_get_num_threads(), {});
       llvm::Value * callThreadNum = Bld.CreateCall(Get_thread_num(), {});

       llvm::BasicBlock * IfInExcess = llvm::BasicBlock::Create(
           CGM.getLLVMContext(), ".if.in.excess", CGF.CurFn);

       llvm::BasicBlock * NotInExcess = llvm::BasicBlock::Create(
           CGM.getLLVMContext(), ".not.in.excess", CGF.CurFn);

       llvm::Value * AmIInExcess = Bld.CreateICmpUGE(callThreadNum, NumThreads);
       Bld.CreateCondBr(AmIInExcess, IfInExcess, NotInExcess);

       // if it is in excess, just go back to syncthreads
       Bld.SetInsertPoint(IfInExcess);
       Bld.CreateBr(CheckFinished);

       // else, do the parallel
       Bld.SetInsertPoint(NotInExcess);
     }

     PushNewParallelRegion(CurrentIsNested);
   }

   llvm::Value * CallParallelRegionPrepare(CodeGenFunction &CGF) {
     llvm::Value * call = CGF.EmitRuntimeCall(OPENMPRTL_FUNC(
         kernel_prepare_parallel));
     return call;
   }

   void CallParallelRegionStart(CodeGenFunction &CGF) {
       CGF.EmitRuntimeCall(OPENMPRTL_FUNC(kernel_parallel));
   }

   void CallParallelRegionEnd(CodeGenFunction &CGF) {
        CGF.EmitRuntimeCall(OPENMPRTL_FUNC(kernel_end_parallel));
   }

   void CallSerializedParallelStart(CodeGenFunction &CGF) {
     llvm::Value *RealArgs[] = {
         CreateIntelOpenMPRTLLoc(clang::SourceLocation(), CGF, 0),
         CreateOpenMPGlobalThreadNum(clang::SourceLocation(), CGF)};
       CGF.EmitRuntimeCall(OPENMPRTL_FUNC(serialized_parallel), RealArgs);
   }

   void CallSerializedParallelEnd(CodeGenFunction &CGF) {
     llvm::Value *RealArgs[] = {
         CreateIntelOpenMPRTLLoc(clang::SourceLocation(), CGF, 0),
         CreateOpenMPGlobalThreadNum(clang::SourceLocation(), CGF)};
     CGF.EmitRuntimeCall(OPENMPRTL_FUNC(end_serialized_parallel),
         RealArgs);
   }

   // the following function disables the barrier after firstprivate, reduction
   // and copyin. This is not needed on nvptx backend because the control loop
   // semantics forces us to do a barrier at the end, no matter if the user
   // specified nowait
   bool RequireFirstprivateSynchronization() { return false; }

   // the following two functions deal with nested parallelism
   // by calling the appropriate codegen functions above
   void EnterParallelRegionInTarget(CodeGenFunction &CGF,
                                    OpenMPDirectiveKind DKind,
                                    ArrayRef<OpenMPDirectiveKind> SKinds,
                                    const OMPExecutableDirective &S) {
     CGBuilderTy &Bld = CGF.Builder;

     OMPRegionTypesStack.push_back(OMP_Parallel);

     if (!NestedParallelStack.back()) { // not already in a parallel region

       // clear up the data structure that will be used to determine the
       // optimal amount of simd lanes to be used in this region
       SimdAndWorksharingNesting.reset();

       // now done after codegen for #parallel region
       // analyze parallel region and calculate best number of lanes
       llvm::Instruction *LoadSimdNumLanes = Bld.CreateLoad(SimdNumLanes);

       // remember insert point to set optimal number of lanes after codegen
       // for the #parallel region
       OptimalNumLanesSetPoint = LoadSimdNumLanes;

       llvm::Value *PrepareParallelArgs[] = {
           Bld.CreateCall(Get_num_threads(), {}), LoadSimdNumLanes};

       llvm::CallInst *PrepareParallel = CGF.EmitRuntimeCall(
           OPENMPRTL_FUNC(kernel_prepare_parallel), PrepareParallelArgs);

       Bld.CreateStore(PrepareParallel, CudaThreadsInParallel);

       CGM.getOpenMPRuntime().GenerateNextLabel(CGF, false, true);

       // Increment the nesting level
       Bld.CreateStore(
           Bld.CreateAdd(Bld.CreateLoad(ParallelNesting), Bld.getInt32(1)),
           ParallelNesting);

       // check if thread does not act either as a lane or as a thread (called
       // excluded from parallel region)
       llvm::Value *MyThreadId = Bld.CreateCall(Get_thread_num(), {});
       llvm::Value *AmINotInParallel =
           Bld.CreateICmpSGE(MyThreadId, Bld.CreateLoad(CudaThreadsInParallel));

       llvm::BasicBlock *IfIsNoLaneNoParallelThread = llvm::BasicBlock::Create(
           CGM.getLLVMContext(), ".if.is.excluded", CGF.CurFn);

       llvm::BasicBlock *IfIsParallelThreadOrLane = llvm::BasicBlock::Create(
           CGM.getLLVMContext(), ".if.is.parthread.or.lane", CGF.CurFn);

       Bld.CreateCondBr(AmINotInParallel, IfIsNoLaneNoParallelThread,
                        IfIsParallelThreadOrLane);

       Bld.SetInsertPoint(IfIsNoLaneNoParallelThread);

       // this makes sure no extra thread that was started by a kernel
       // will participate in the parallel region, including simd or
       // nested parallelism
       Bld.CreateStore(Bld.CreateLoad(SimdNumLanes), SimdLaneNum);

       Bld.CreateBr(SynchronizeAndNextState);

       Bld.SetInsertPoint(IfIsParallelThreadOrLane);

       // calculate my simd lane num to exclude cuda threads that will
       // only act as simd lanes and not parallel threads
       Bld.CreateStore(Bld.CreateAnd(Bld.CreateCall(Get_thread_num(), {}),
                                     Bld.CreateSub(Bld.CreateLoad(SimdNumLanes),
                                                   Bld.getInt32(1))),
                       SimdLaneNum);

       llvm::Value *InitParallelArgs[] = {Bld.CreateLoad(SimdNumLanes)};

       CGF.EmitRuntimeCall(OPENMPRTL_FUNC(kernel_parallel), InitParallelArgs);

       // only lane id 0 (lane master) is a thread in parallel

       llvm::BasicBlock *ParallelRegionCG = llvm::BasicBlock::Create(
           CGM.getLLVMContext(), ".par.reg.code", CGF.CurFn);

       Bld.CreateCondBr(
           Bld.CreateICmpNE(Bld.CreateLoad(SimdLaneNum), Bld.getInt32(0)),
           SynchronizeAndNextState, ParallelRegionCG);

       Bld.SetInsertPoint(ParallelRegionCG);

     } else { // nested parallel region: serialize!
       CallSerializedParallelStart (CGF);
     }

     PushNewParallelRegion(true);
   }

   void ExitParallelRegionInTarget(CodeGenFunction &CGF) {
     CGBuilderTy &Bld = CGF.Builder;
     // Decrement the nesting level
     Bld.CreateStore(
         Bld.CreateSub(Bld.CreateLoad(ParallelNesting), Bld.getInt32(1)),
         ParallelNesting);

     assert((OMPRegionTypesStack.back() == OMP_Parallel) &&
            "Exiting a parallel region does not match stack state");
     OMPRegionTypesStack.pop_back();

     // we need to inspect the previous layer to understand what type
     // of end we need
     PopParallelRegion();
     // check if we are in a nested parallel region
     if (!NestedParallelStack.back()) { // not nested parallel
       // we are now able to determine the optimal amount of lanes to be
       // used in this #parallel region and add the amount setting in the right
       // place, just before we start the region
       int OptimalNumLanes = CalculateNumLanes();
       llvm::Instruction *StoreOptimalLanes =
           new llvm::StoreInst(Bld.getInt32(OptimalNumLanes), SimdNumLanes);
       OptimalNumLanesSetPoint->getParent()->getInstList().insert(
           OptimalNumLanesSetPoint, StoreOptimalLanes);

       // signal runtime that we are closing the parallel region and
       // switch to new team-sequential label
       CallParallelRegionEnd(CGF);
       CGM.getOpenMPRuntime().GenerateNextLabel(CGF, true, false);

       // update the global target optimal number of simd lanes to be used
       // with information from this: currently calculate maximum over all
       // parallel regions
       int8_t CurrentOptimalSimdLanes =
           (GetNumSimdLanesPerTargetRegion() < OptimalNumLanes)
               ? OptimalNumLanes
               : GetNumSimdLanesPerTargetRegion();
       SetNumSimdLanesPerTargetRegion(CurrentOptimalSimdLanes);
     } else { // nested parallel region: close serialize
       CallSerializedParallelEnd (CGF);
     }
   }

   void SupportCritical (const OMPCriticalDirective &S, CodeGenFunction &CGF,
       llvm::Function * CurFn, llvm::GlobalVariable *Lck) {
     CGBuilderTy &Builder = CGF.Builder;
     llvm::Value *Loc = CreateIntelOpenMPRTLLoc(S.getLocStart(), CGF, 0);


       //  OPENMPRTL_LOC(S.getLocStart(), CGF);
     llvm::Value *GTid = Builder.CreateCall(Get_thread_num(), {});
     llvm::Value *RealArgs[] = { Loc, GTid, Lck };

     llvm::BasicBlock * preLoopBlock = Builder.GetInsertBlock();
     llvm::BasicBlock * criticalLoopBlock =
         llvm::BasicBlock::Create(CGM.getLLVMContext(), ".critical.loop",
             CurFn);
       llvm::BasicBlock * criticalExecBlock =
           llvm::BasicBlock::Create(CGM.getLLVMContext(), ".critical.exec",
               CurFn);
       llvm::BasicBlock * criticalSkipBlock =
           llvm::BasicBlock::Create(CGM.getLLVMContext(), ".critical.skip");
       llvm::BasicBlock * criticalLoopEndBlock =
           llvm::BasicBlock::Create(CGM.getLLVMContext(), ".critical.loop.end");
       llvm::Value *laneIndex = llvm::CastInst::CreateZExtOrBitCast(
           Builder.CreateAnd(GTid,0x1f),llvm::Type::getInt64Ty(
               CGM.getLLVMContext()),"laneIndex",preLoopBlock);
       Builder.CreateBr(criticalLoopBlock);
       Builder.SetInsertPoint(criticalLoopBlock);
       llvm::PHINode *loopiv = Builder.CreatePHI(llvm::Type::getInt64Ty(
           CGM.getLLVMContext()),2,"critical_loop_iv");
       llvm::Value *init = llvm::ConstantInt::get(llvm::Type::getInt64Ty(
           CGM.getLLVMContext()),0);
       loopiv->addIncoming(init,preLoopBlock);
       llvm::Value *myturn = Builder.CreateICmpEQ(laneIndex,loopiv,"myturn");
       Builder.CreateCondBr(myturn,criticalExecBlock,criticalSkipBlock);
       Builder.SetInsertPoint(criticalExecBlock);
       CGF.EmitRuntimeCall(OPENMPRTL_FUNC(critical), RealArgs);
       CGF.EmitOMPCapturedBodyHelper(S);
       CGF.EmitRuntimeCall(OPENMPRTL_FUNC(end_critical), RealArgs);
       Builder.CreateBr(criticalSkipBlock);
       CurFn->getBasicBlockList().push_back(criticalSkipBlock);
       Builder.SetInsertPoint(criticalSkipBlock);
       llvm::Value *bump = llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()),1);
       llvm::Value *bumpedIv = Builder.CreateAdd(loopiv,bump,"bumpediv");
       loopiv->addIncoming(bumpedIv,criticalSkipBlock);
       //llvm::Value *limit = llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()),32);
       //llvm::Value *finished = Builder.CreateICmpULT(bumpedIv,limit,"finished");
       llvm::Value *limit = llvm::ConstantInt::get(llvm::Type::getInt64Ty(CGM.getLLVMContext()),31);
       llvm::Value *finished = Builder.CreateICmpULT(limit,bumpedIv,"finished");
       Builder.CreateCondBr(finished,criticalLoopEndBlock,criticalLoopBlock);
       CurFn->getBasicBlockList().push_back(criticalLoopEndBlock);
       Builder.SetInsertPoint(criticalLoopEndBlock);
   }

   void EmitNativeBarrier(CodeGenFunction &CGF) {
     CGBuilderTy &Bld = CGF.Builder;

     Bld.CreateCall(Get_syncthreads(), {});
   }

   // #pragma omp simd specialization for NVPTX
   // \warning assume no more than 32 lanes in #simd
   void EmitSimdInitialization(llvm::Value *LoopIndex, llvm::Value *LoopCount,
                               CodeGenFunction &CGF) {

     // sequential behavior in case of reduction clause detected
     if (SimdHasReduction) {
       CGOpenMPRuntime::EmitSimdInitialization(LoopIndex, LoopCount, CGF);
       return;
     }

     CGBuilderTy &Builder = CGF.Builder;

     llvm::Value *SimdLaneNumSext =
         Builder.CreateSExt(SimdLaneNum, LoopCount->getType()->getPointerTo());

     llvm::Value *InitialValue =
         Builder.CreateAdd(llvm::ConstantInt::get(LoopCount->getType(), 0),
                           Builder.CreateLoad(SimdLaneNumSext));
     Builder.CreateStore(InitialValue, LoopIndex);
   }

   void EmitSimdIncrement(llvm::Value *LoopIndex, llvm::Value *LoopCount,
                          CodeGenFunction &CGF) {

     // sequential behavior in case of reduction clause detected
     if (SimdHasReduction) {
       CGOpenMPRuntime::EmitSimdIncrement(LoopIndex, LoopCount, CGF);
       return;
     }

     CGBuilderTy &Builder = CGF.Builder;

     llvm::Value *NewLoopIndexValue = Builder.CreateAdd(
         Builder.CreateLoad(LoopIndex), Builder.CreateLoad(SimdNumLanes));

     Builder.CreateStore(NewLoopIndexValue, LoopIndex);
   }

   void StartNewTargetRegion() {
     // reset some class instance variables for a new target region
     MasterLabelShared = 0;
     OthersLabelShared = 0;
     regionLabelMap.clear();
     NextId = 0;
     //     InspectorExecutorSwitch = 0;
     StartControl = 0;
     EndControl = 0;
     FinishedVar = 0;
     CheckFinished = 0;
     SequentialStartBlock = 0;

     SimdLaneNum = 0;
     NextState = 0;
     ControlStateIndex = 0;
     SynchronizeAndNextState = 0;
     SimdNumLanes = 0;
     ControlState = 0;
     CudaThreadsInParallel = 0;
     EndTarget = 0;
     FinishedCase = 0;

     // retire this stack, use the one below
     NestedParallelStack.clear();
     PushNewParallelRegion(false); // we start in a sequential region

     // start with initial target, add teams if needed when encountered
     OMPRegionTypesStack.clear();
     OMPRegionTypesStack.push_back(OMP_InitialTarget);
     SimdAndWorksharingNesting.reset();
     NextBitSimdAndWorksharingNesting = 0;

     // reset to 1 for new target region
     NumSimdLanesPerTargetRegion = 1;

     // each target region has a thread limit global variable: reset to
     // guarantee
     // it is created
     ThreadLimitGlobal = 0;
   }

   void StartTeamsRegion() {
     // a teams construct always start with a team master-only region
     OMPRegionTypesStack.push_back(OMP_TeamSequential);

     // no need to close it at the end: by OMP specifications, teams pragma
     // has to be closely nested inside target and no statement can be outside
     // of it in a target region when it has a teams region
   }

public:
  unsigned GLOBAL_ADDRESS_SPACE = 1;
   unsigned SHARED_ADDRESS_SPACE = 3;

   CGOpenMPRuntime_NVPTX(CodeGenModule &CGM)
       : CGOpenMPRuntime(CGM),
         ArchName(CGM.getTarget().getTriple().getArchName()), MASTER_ID(0),
         MasterLabelShared(0), OthersLabelShared(0), NextId(0), StartControl(0),
         EndControl(0), FinishedVar(0), SequentialStartBlock(0),
         CheckFinished(0), FinishedCase(0), SynchronizeAndNextState(0),
         EndTarget(0), NextBitSimdAndWorksharingNesting(0),
         NumSimdLanesPerTargetRegion(1), OptimalNumLanesSetPoint(0),
         ParallelNesting(0), NextState(0), ControlState(0),
         ControlStateIndex(0), CudaThreadsInParallel(0), SimdNumLanes(0),
         SimdLaneNum(0), ControlSwitch(0), SimdHasReduction(false),
         ThreadLimitGlobal(0) {

     SimdAndWorksharingNesting.resize(EXPECTED_WS_NESTS);

     // FIXME: Make this depend on some compiler options and pick some better
     // default values.
     SharedStackType = CGM.getLangOpts().OpenMPNVPTXFastShare
                           ? SharedStackType_Fast
                           : SharedStackType_Default;
     SharedStackDynamicAlloc = false;
     assert(CGM.getLangOpts().OMPNVPTXSharingSizesPerThread.size() >= 2 &&
            "Unexpected shared size default values");
     SharedStackSizePerThread[0] =
         CGM.getLangOpts().OMPNVPTXSharingSizesPerThread[0];
     SharedStackSizePerThread[1] =
         CGM.getLangOpts().OMPNVPTXSharingSizesPerThread[1];
     SharedStackSizePerTeam = CGM.getLangOpts().OMPNVPTXSharingSizePerTeam;
     SharedStackSize = CGM.getLangOpts().OMPNVPTXSharingSizePerKernel;

     LocalThrTy = llvm::StructType::create(
         "local_thr_info", CGM.Int32Ty /* priv */,
         CGM.Int32Ty /* current_event */, CGM.Int32Ty /* eventsNumber */,
         CGM.Int32Ty /* chunk_warp */, CGM.Int32Ty /* num_iterations */, NULL);
  }

  /// Implement some target dependent transformation for the target region
  /// outlined function
  ///
  virtual void PostProcessTargetFunction(llvm::Function *F) {

    CGOpenMPRuntime::PostProcessTargetFunction(F);

    // No further post processing required if we are not in target mode
    if (!CGM.getLangOpts().OpenMPTargetMode)
      return;

    llvm::Module &M = CGM.getModule();
    llvm::LLVMContext &C = M.getContext();

    // Get "nvvm.annotations" metadata node
    llvm::NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

    llvm::Metadata *MDVals[] = {
        llvm::ConstantAsMetadata::get(F), llvm::MDString::get(C, "kernel"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 1))};
    // Append metadata to nvvm.annotations
    MD->addOperand(llvm::MDNode::get(C, MDVals));
  }

  void PostProcessPrintfs(llvm::Module &M) {
    llvm::Function *PrintfFunc = FindPrintfFunction(M);

    if (PrintfFunc == nullptr) {
      return;
    }

    llvm::Function *VprintfFunc = InsertVprintfDeclaration(M);
    const llvm::DataLayout &DL = M.getDataLayout();

    // Go over all the uses of printf in the module. The iteration pattern here
    // (increment the iterator immediately after grabbing the current
    // instruction) is required to allow this loop to remove the actual uses
    // and still keep running over all of them properly.
    for (llvm::Value::use_iterator UI = PrintfFunc->use_begin(),
                                   UE = PrintfFunc->use_end();
         UI != UE;) {
      llvm::CallInst *Call = dyn_cast<llvm::CallInst>(UI->getUser());
      if (!Call) {
        llvm::report_fatal_error(
            "Only 'call' uses of 'printf' are allowed for NVPTX");
      }
      UI++;

      // First compute the buffer size required to hold all the formatting
      // arguments, and create the buffer with an alloca.
      // Note: the first argument is the formatting string - its validity is
      // verified by the frontend.
      unsigned BufSize = 0;
      for (unsigned I = 1, IE = Call->getNumArgOperands(); I < IE; ++I) {
        llvm::Value *Operand = Call->getArgOperand(I);
        BufSize = llvm::RoundUpToAlignment(
            BufSize, DL.getPrefTypeAlignment(Operand->getType()));
        BufSize += DL.getTypeAllocSize(Call->getArgOperand(I)->getType());
      }

      llvm::Type *Int32Ty = llvm::Type::getInt32Ty(M.getContext());
      llvm::Value *BufferPtr = nullptr;

      if (BufSize == 0) {
        // If no arguments, pass an empty buffer as the second argument to
        // vprintf.
        BufferPtr = new llvm::AllocaInst(llvm::Type::getInt8Ty(M.getContext()),
                                         llvm::ConstantInt::get(Int32Ty,
                                                                BufSize),
                                         "buf_for_vprintf_args", Call);
      } else {
        // Create the buffer to hold all the arguments. Align it to the
        // preferred alignment of the first object going into the buffer.
        // Note: if BufSize > 0, we know there's at least one object so
        // getArgOperand(1) is safe.
        unsigned AlignOfFirst =
            DL.getPrefTypeAlignment(Call->getArgOperand(1)->getType());
        llvm::Type *PointeeType = llvm::Type::getInt8Ty(M.getContext());
        BufferPtr = new llvm::AllocaInst(PointeeType,
                                         llvm::ConstantInt::get(Int32Ty,
                                                                BufSize),
                                         AlignOfFirst,
                                         "buf_for_vprintf_args", Call);

        // Each argument is placed into the buffer as follows:
        // 1. GEP is used to compute an offset into the buffer
        // 2. Bitcast to convert the buffer pointer to the correct type
        // 3. Store into that location
        unsigned Offset = 0;
        for (unsigned I = 1, IE = Call->getNumArgOperands(); I < IE; ++I) {
          llvm::Value *Operand = Call->getArgOperand(I);
          Offset = llvm::RoundUpToAlignment(
              Offset, DL.getPrefTypeAlignment(Operand->getType()));

          llvm::GetElementPtrInst *GEP = llvm::GetElementPtrInst::Create(
              PointeeType, BufferPtr, llvm::ConstantInt::get(Int32Ty, Offset),
              "", Call);

          llvm::BitCastInst *Cast =
              new llvm::BitCastInst(
                  GEP, Operand->getType()->getPointerTo(), "", Call);
          new llvm::StoreInst(Operand, Cast, false,
                        DL.getPrefTypeAlignment(Operand->getType()), Call);

          Offset += DL.getTypeAllocSize(Operand->getType());
        }
      }

      // Generate the alternative call to vprintf and replace the original.
      llvm::Value *VprintfArgs[] = {Call->getArgOperand(0), BufferPtr};
      llvm::CallInst *VprintfCall =
          llvm::CallInst::Create(VprintfFunc, VprintfArgs, "", Call);

      Call->replaceAllUsesWith(VprintfCall);
      Call->eraseFromParent();
    }    
  }

  llvm::Function *FindPrintfFunction(llvm::Module &M) {
    // Looking for a declaration of a function named "printf". If this function
    // is *defined* in the module, bail out.
    llvm::Function *PrintfFunc = M.getFunction("printf");
    if (!PrintfFunc || !PrintfFunc->isDeclaration())
      return nullptr;

    // So this is just a declaration. If so, it must match what we expect from
    // printf; otherwise, it's an error.
    llvm::FunctionType *FT = PrintfFunc->getFunctionType();

    if (FT->getNumParams() == 1 && FT->isVarArg() &&
        FT->getReturnType() == llvm::Type::getInt32Ty(M.getContext()) &&
        FT->getParamType(0) == llvm::Type::getInt8PtrTy(M.getContext())) {
      return PrintfFunc;
    } else {
      llvm::report_fatal_error(
          "Found printf in module but it has an invalid type");
      return nullptr;
    }
  }

  llvm::Function *InsertVprintfDeclaration(llvm::Module &M) {
    if (M.getFunction("vprintf") != nullptr) {
      llvm::report_fatal_error(
          "It is illegal to declare vprintf with C linkage");
    }

    // Create a declaration for vprintf with the proper type and insert it into
    // the module.
    llvm::Type *ArgTypes[] = {llvm::Type::getInt8PtrTy(M.getContext()),
                              llvm::Type::getInt8PtrTy(M.getContext())};
    llvm::FunctionType *VprintfFuncType =
        llvm::FunctionType::get(llvm::Type::getInt32Ty(
            M.getContext()), ArgTypes, false);

    return llvm::Function::Create(VprintfFuncType,
                                  llvm::GlobalVariable::ExternalLinkage,
                                  "vprintf", &M);
  }

  virtual llvm::Value *CreateIntelOpenMPRTLLoc(SourceLocation Loc,
      CodeGenFunction &CGF, unsigned Flags) {
    //The Loc struct is not used by the target therefore we do not perform
    //any initialization

    return CGF.CreateTempAlloca(
  		llvm::IdentTBuilder::get(CGM.getLLVMContext()));
  }

  virtual llvm::Value *CreateOpenMPGlobalThreadNum(SourceLocation Loc,
      CodeGenFunction &CGF) {

    //FIXME: Not sure this is what we want, I am computing global thread ID
    //as blockID*BlockSize * threadID

    llvm::Value *BId = CGF.Builder.CreateCall(Get_team_num(), {}, "blockid");
    llvm::Value *BSz = CGF.Builder.CreateCall(Get_num_threads(), {}, "blocksize");
    llvm::Value *TId = CGF.Builder.CreateCall(Get_thread_num(), {}, "threadid");

    return CGF.Builder.CreateAdd(CGF.Builder.CreateMul(BId, BSz), TId, "gid");
  }

  // special handling for composite pragmas:
  bool RequiresStride(OpenMPDirectiveKind Kind, OpenMPDirectiveKind SKind) {
    switch (Kind) {
    case OMPD_for_simd:
    case OMPD_parallel_for_simd:
      return true;
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
      if (SKind == OMPD_distribute) {
        return true;
      }
    default:
      break;
    }
    return false;
  }

  llvm::Value * GetNextIdIncrement(CodeGenFunction &CGF,
		  bool IsStaticSchedule, const Expr * ChunkSize, llvm::Value * Chunk,
		  llvm::Type * IdxTy, QualType QTy, llvm::Value * Idx,
		  OpenMPDirectiveKind Kind, OpenMPDirectiveKind SKind, llvm::Value * PSt) {

    CGBuilderTy &Builder = CGF.Builder;
    llvm::Value *NextIdx;

    // when distribute contains a parallel for, each distribute iteration
    // executes "stride" instructions of the innermost for
    // also valid for #for simd, because we explicitly transform the
    // single loop into two loops

    if (RequiresStride(Kind, SKind)) {
      llvm::Value *Stride = Builder.CreateLoad(PSt);
      NextIdx = Builder.CreateAdd(Idx, Stride, ".next.idx.", false,
                                  QTy->isSignedIntegerOrEnumerationType());
    } else
      NextIdx =
          Builder.CreateAdd(Idx, llvm::ConstantInt::get(IdxTy, 1), ".next.idx.",
                            false, QTy->isSignedIntegerOrEnumerationType());

    assert(NextIdx && "NextIdx variable not set");

          return NextIdx;
  }

  // Insert the overload of the default kmpc calls' implementation here, e.g.:
  //
  // TARGET_EMIT_OPENMP_FUNC(
  //    <name of the kmpc call> ,
  //    <body of the function generation - Fn is the current function and Bld
  //    is the builder for the the entry basic block>

  // ...or specialize it by hand as we do f for fork_call and fork_teams
  llvm::Constant* Get_fork_call(){
    llvm::Function *Fn = cast<llvm::Function>(CGM.CreateRuntimeFunction(
            llvm::TypeBuilder<__kmpc_fork_call, false>::get(
            		CGM.getLLVMContext()),
					(Twine("__kmpc_",ArchName) + Twine("fork_call")).str()));

    llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(CGM.getLLVMContext(), "entry", Fn);
    CGBuilderTy Bld(EntryBB);
    {
      assert(Fn->arg_size() == 4 && "Unexpected number of arguments");

	  // the helper function is inlined - it is just a function call
	  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);
	  Fn->addFnAttr(llvm::Attribute::AlwaysInline);

	  llvm::Function::arg_iterator arg = Fn->arg_begin();
	  std::advance(arg, 2);  // get to the function we need calling
	  llvm::Value *BitCastedFunction = arg;
	  std::advance(arg, 1); // arguments of function to be called
	  llvm::Value *FunctionArgs = arg;

	  SmallVector<llvm::Type *, 3> FnArgTypes;
	  FnArgTypes.push_back(CGM.Int32Ty->getPointerTo());
	  FnArgTypes.push_back(CGM.Int32Ty->getPointerTo());
	  FnArgTypes.push_back(CGM.Int8Ty->getPointerTo());
	  llvm::FunctionType * FnTy =
		  llvm::FunctionType::get(CGM.VoidTy, FnArgTypes, false);

	  llvm::AllocaInst * gtidEmpty = Bld.CreateAlloca(
		  Bld.getInt32Ty());
	  llvm::AllocaInst * boundEmpty = Bld.CreateAlloca(
						Bld.getInt32Ty());

	  llvm::Value * BitCastedBackFunction = Bld.CreateBitCast(
		  BitCastedFunction, FnTy->getPointerTo());

	  llvm::Value * BitCastedArgs = Bld.CreateBitCast(FunctionArgs,
		  CGM.Int8Ty->getPointerTo());

	  // For target nvptx we pass 0s as global thread id and thread id
	  // these values can be retrieved from the thread's own state instead
	  // of having them in the function parameters
	  llvm::Value * RealArgs[] = { gtidEmpty, boundEmpty, BitCastedArgs };

	  // emit a call to the microtask function using the passed args
	  Bld.CreateCall(BitCastedBackFunction, makeArrayRef(RealArgs));

	  // Unset the number of threads required by the parallel region at the end
      llvm::Function *UnsetFn = cast<llvm::Function>(CGM.CreateRuntimeFunction(
              llvm::TypeBuilder<__kmpc_unset_num_threads, false>::get(
              		CGM.getLLVMContext()),
  					"__kmpc_unset_num_threads"));
      Bld.CreateCall(UnsetFn, {});

	  Bld.CreateRetVoid();
    }
    return Fn;
  }

  llvm::Constant* Get_fork_teams(){
    llvm::Function *Fn = cast<llvm::Function>(CGM.CreateRuntimeFunction(
            llvm::TypeBuilder<__kmpc_fork_teams, false>::get(
            		CGM.getLLVMContext()),
					(Twine("__kmpc_",ArchName) + Twine("fork_teams")).str()));
    llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(CGM.getLLVMContext(), "entry", Fn);
    CGBuilderTy Bld(EntryBB);
    {
      assert(Fn->arg_size() == 4 && "Unexpected number of arguments");

      // the helper function is inlined - it is just a function call
      Fn->setLinkage(llvm::GlobalValue::InternalLinkage);
      Fn->addFnAttr(llvm::Attribute::AlwaysInline);

      llvm::Function::arg_iterator arg = Fn->arg_begin();
      std::advance(arg, 2);  // get to the function we need calling
      llvm::Value *BitCastedFunction = arg;
      std::advance(arg, 1); // arguments of function to be called
      llvm::Value *FunctionArgs = arg;

      SmallVector<llvm::Type *, 3> FnArgTypes;
      FnArgTypes.push_back(CGM.Int32Ty->getPointerTo());
      FnArgTypes.push_back(CGM.Int32Ty->getPointerTo());
      FnArgTypes.push_back(CGM.Int8Ty->getPointerTo());
      llvm::FunctionType * FnTy =
    		  llvm::FunctionType::get(CGM.VoidTy, FnArgTypes, false);

      llvm::AllocaInst * gtidEmpty = Bld.CreateAlloca(
    		  Bld.getInt32Ty());
      llvm::AllocaInst * boundEmpty = Bld.CreateAlloca(
    		  Bld.getInt32Ty());

      llvm::Value * BitCastedBackFunction = Bld.CreateBitCast(
    		  BitCastedFunction, FnTy->getPointerTo());

      llvm::Value * BitCastedArgs = Bld.CreateBitCast(FunctionArgs,
    		  CGM.Int8Ty->getPointerTo());

      // For target nvptx we pass 0s as global thread id and thread id
      // these values can be retrieved from the thread's own state
      // instead of having them in the function parameters
      llvm::Value * RealArgs[] = {gtidEmpty, boundEmpty, BitCastedArgs};

      // emit a call to the microtask function using the passed args
      Bld.CreateCall(BitCastedBackFunction, makeArrayRef(RealArgs));
      Bld.CreateRetVoid();
    }
    return Fn;
  }

  llvm::Value * AllocateThreadLocalInfo(CodeGenFunction & CGF) {
	    CGBuilderTy &Bld = CGF.Builder;

	    return Bld.CreateAlloca(LocalThrTy);
  }

  // these are run-time functions which are only exposed by the gpu library
  typedef void(__kmpc_unset_num_threads)();

  bool requiresMicroTaskForTeams(){
    return false;
  }
  bool requiresMicroTaskForParallel(){
    return false;
  }

  /// NVPTX targets cannot take advantage of the entries ordering to retrieve
  /// symbols, therefore we need to rely on names. We are currently failing
  /// if this target is being used as host because the linker cannot combine
  /// the entries in the same section as desired and do not generate any symbols
  /// in target mode (we just can't use them)

  llvm::GlobalVariable *CreateHostPtrForCurrentTargetRegion(const Decl *D,
                                                            llvm::Function *Fn,
                                                            StringRef Name) {
    if (CGM.getLangOpts().OpenMPTargetMode)
      return nullptr;

    llvm_unreachable("This target cannot be used as OpenMP host");
    return nullptr;
  }

  llvm::GlobalVariable *CreateHostEntryForTargetGlobal(const Decl *D,
                                                       llvm::GlobalVariable *GV,
                                                       StringRef Name) {

    if (CGM.getLangOpts().OpenMPTargetMode) {

      const VarDecl *VD = cast<VarDecl>(D);

      // Create an externally visible global variable for static data so it can
      // be loaded by the OpenMP runtime
      if (VD->getStorageClass() == SC_Static) {
        StaticEntries.insert(GV);
      }
      return nullptr;
    }

    llvm_unreachable("This target cannot be used as OpenMP host");
    return nullptr;
  }

  /// This is a hook to enable postprocessing of the module. By default this
  /// only does the creation of globals from local variables due to data sharing
  /// constraints
  void PostProcessModule(CodeGenModule &CGM) {

    if (ValuesToBeInSharedMemory.size()){
      // We need to use a shared address space in order to share data between
      // threads. This data is going to be stored in the form of a stack and
      // currently support 2 nesting levels.

      // Create storage for the shared data based on the information passed by
      // the user.
      llvm::GlobalVariable *SharedData = nullptr;

      {
        uint64_t Size = 0;
        uint64_t AddrSpace = 0;
        switch (SharedStackType) {
        default:
          // The fast version relies on global memory, so we need to allocate
          // storage for all teams (blocks)
          Size = SharedStackSize;
          AddrSpace = GLOBAL_ADDRESS_SPACE;
          break;
        case SharedStackType_Fast:
          // The fast version relies on shared memory, so we only need to
          // allocate
          // storage per team
          Size = SharedStackSizePerTeam;
          AddrSpace = SHARED_ADDRESS_SPACE;
          break;
        }

        llvm::ArrayType *SharedDataTy = llvm::ArrayType::get(
            llvm::Type::getInt8Ty(CGM.getLLVMContext()), Size);
        SharedData = new llvm::GlobalVariable(
            CGM.getModule(), SharedDataTy, false,
            llvm::GlobalValue::CommonLinkage,
            llvm::Constant::getNullValue(SharedDataTy),
            Twine("__omptgt__shared_data_"), 0,
            llvm::GlobalVariable::NotThreadLocal, AddrSpace, false);
      }

      // Look in all the sharing regions and replace local variables with shared
      // ones if needed.
      for (auto &Region : ValuesToBeInSharedMemory) {
        // If no data was registered for this region, just move to the next one
        if (Region.empty())
          continue;

        // Scan the different levels. We only parallelize up to the second level
        // of nesting.
        for (unsigned LevelIdx = 0; LevelIdx < 2; ++LevelIdx) {

          // We don't have more levels in this regions, so lets move forward to
          // the next one.
          if (Region.size() <= LevelIdx)
            break;

          auto &Sets = Region[LevelIdx];

          if (Sets.empty())
            continue;

          // Separate VLA from everything else as we need to special case for
          // them
          llvm::SmallVector<llvm::AllocaInst *, 32> FLAlloca;
          llvm::SmallVector<llvm::ConstantInt *, 32> FLAllocaSizes;
          llvm::SmallVector<llvm::AllocaInst *, 32> VLAlloca;
          llvm::SmallVector<llvm::Value *, 32> VLAllocaSizes;

          llvm::SmallVector<llvm::LoadInst *, 32> VLASizeLoads;

          for (auto &Vars : Sets) {

            if (Vars.empty())
              continue;

            for (auto V : Vars) {

              if (llvm::LoadInst *L = dyn_cast<llvm::LoadInst>(V)) {
                VLASizeLoads.push_back(L);
                continue;
              }

              llvm::AllocaInst *AI = cast<llvm::AllocaInst>(V);
              llvm::Value *ArraySize = AI->getArraySize();

              if (llvm::ConstantInt *CI =
                      dyn_cast<llvm::ConstantInt>(ArraySize)) {
                FLAlloca.push_back(AI);
                if (AI->isArrayAllocation())
                  FLAllocaSizes.push_back(CI);
                else
                  FLAllocaSizes.push_back(nullptr);
              } else {
                VLAlloca.push_back(AI);

                // We are expecting to get here only variable size arrays
                assert(AI->isArrayAllocation() &&
                       "Expecting only arrays here!");
                VLAllocaSizes.push_back(AI->getArraySize());
              }
            }
          }

          // If we don't have anything to share lets look at the next level
          if (FLAlloca.empty() && VLAlloca.empty())
            continue;

          // Create the type that accommodates all the data for this level. For
          // VLAs we use a pointer to the place where the array is instead. So
          // if we are sharing int a, int b, int c[n], int d[m] the stack will
          // have:
          // //static sizes
          // int a
          // int b
          // int *pc = &c[0]
          // int *pd = &d[0]
          // //dynamic sizes
          // int c[n]
          // int d[n]
          // The last two are not parted of the struct and their indexing is
          // controlled dynamically by the stack pointer created below.

          llvm::StructType *LevelTy;
          {
            llvm::SmallVector<llvm::Type *, 32> Tys;
            for (unsigned i = 0; i < FLAlloca.size(); ++i) {
              llvm::AllocaInst *AI = FLAlloca[i];

              // If we are in dynamic mode we use mallocs to create the storage
              // and use the address directly here.
              if (SharedStackDynamicAlloc) {
                Tys.push_back(AI->getType());
                continue;
              }

              // If this is an array, we need to take its size into account
              if (llvm::ConstantInt *C = FLAllocaSizes[i]) {
                Tys.push_back(llvm::ArrayType::get(AI->getAllocatedType(),
                                                   C->getSExtValue()));
                continue;
              }

              Tys.push_back(AI->getAllocatedType());
            }
            for (unsigned i = 0; i < VLAlloca.size(); ++i) {
              llvm::AllocaInst *AI = FLAlloca[i];
              Tys.push_back(AI->getType());
            }
            LevelTy = llvm::StructType::create(Tys, ".sharing_struct");
          }

          llvm::PointerType *LevelTyPtr = LevelTy->getPointerTo(
              cast<llvm::PointerType>(SharedData->getType())
                  ->getAddressSpace());

          // For each level we need a variable to trace how the stack grows so
          // all the VLAs get indexed properly (like a stack pointer). If we are
          // in dynamic mode we don't need it as we use malloc and there is no
          // variability in the stack.

          // Get the entry basic block so that we can install the stack pointers
          // in there
          llvm::BasicBlock *EntryBB =
              &(FLAlloca.empty() ? VLAlloca.front() : FLAlloca.front())
                   ->getParent()
                   ->getParent()
                   ->front();
          CGBuilderTy Bld(EntryBB, EntryBB->begin());

          // Compute the initial offset in the storage space where the shared
          // data lives
          llvm::Value *OffsetThd = llvm::ConstantInt::get(CGM.SizeTy, 0);
          llvm::Value *OffsetBlk = llvm::ConstantInt::get(CGM.SizeTy, 0);

          // If the parallelism level is not zero then we need to use an offset
          // that depends on the number of threads
          if (LevelIdx) {
            // Skip level zero storage
            OffsetThd = Bld.CreateAdd(
                OffsetThd, llvm::ConstantInt::get(CGM.SizeTy,
                                                  SharedStackSizePerThread[0]));

            // Add offsets related with the relevant thread (the lane master -
            // the first thread in the 32-thread warp)
            llvm::Value *ThdNum = Bld.CreateIntCast(
                Bld.CreateCall(Get_thread_num(), None), CGM.SizeTy, false);
            llvm::Value *Tmp = Bld.CreateMul(
                ThdNum, llvm::ConstantInt::get(
                            CGM.SizeTy, SharedStackSizePerThread[LevelIdx]));
            OffsetThd = Bld.CreateAdd(OffsetThd, Tmp);
          }

          // If using global memory we also need to add the offset related
          // with blocks
          if (SharedStackType != SharedStackType_Fast) {
            llvm::Value *TeamNum = Bld.CreateIntCast(
                Bld.CreateCall(Get_team_num(), None), CGM.SizeTy, false);
            llvm::Value *TeamOffset = Bld.CreateMul(
                TeamNum,
                llvm::ConstantInt::get(CGM.SizeTy, SharedStackSizePerTeam));
            OffsetBlk = Bld.CreateAdd(OffsetBlk, TeamOffset);
          }

          // Add the size of the struct to the stack pointer so we can start
          // reserving the right size for the VLAs after that.
          llvm::Value *InitialOffset = llvm::ConstantInt::get(CGM.SizeTy, 0);
          InitialOffset = Bld.CreateAdd(InitialOffset, OffsetThd);
          InitialOffset = Bld.CreateAdd(InitialOffset, OffsetBlk);
          InitialOffset = Bld.CreateAdd(
              InitialOffset,
              llvm::ConstantInt::get(
                  CGM.SizeTy,
                  CGM.getModule().getDataLayout().getTypeAllocSize(LevelTy)));

          llvm::AllocaInst *SP =
              Bld.CreateAlloca(CGM.SizeTy, nullptr, ".level_sp");
          Bld.CreateStore(InitialOffset, SP);

          // Get the pointer to the struct that we will use to share data in
          // this level
          auto GetSharedStructPtr = [&](bool CheckLaneMaster) {
            llvm::Value *Offset = llvm::ConstantInt::get(CGM.SizeTy, 0);

            if (!LevelIdx) {
              llvm::Value *InitialOffsetIdx[] = {
                  llvm::ConstantInt::get(CGM.SizeTy, 0), InitialOffset};
              llvm::Value *SharedStructPtr =
                  Bld.CreateGEP(SharedData, InitialOffsetIdx);
              return Bld.CreateBitCast(SharedStructPtr, LevelTyPtr);
            }

            // Skip level zero storage
            Offset = Bld.CreateAdd(
                Offset, llvm::ConstantInt::get(CGM.SizeTy,
                                               SharedStackSizePerThread[0]));

            // Add offsets related with the relevant thread (the lane master -
            // the first thread in the 32-thread warp)
            llvm::Value *ThdNum = Bld.CreateIntCast(
                Bld.CreateCall(Get_thread_num(), None), CGM.SizeTy, false);

            if (CheckLaneMaster) {
              //#ifdef NVPTX_SIMD_IS_WORKING
              llvm::Value *CurrentLevel = Bld.CreateLoad(ParallelNesting);
              llvm::Value *UseSelfSlot = Bld.CreateICmpULT(
                  CurrentLevel,
                  llvm::ConstantInt::get(CurrentLevel->getType(), 2));
              ThdNum = Bld.CreateSelect(
                  UseSelfSlot, ThdNum,
                  Bld.CreateAnd(
                      ThdNum, llvm::ConstantInt::get(CGM.SizeTy, -1ull << 5)));
              //#endif
            }

            llvm::Value *Tmp = Bld.CreateMul(
                ThdNum, llvm::ConstantInt::get(
                            CGM.SizeTy, SharedStackSizePerThread[LevelIdx]));
            Offset = Bld.CreateAdd(Offset, Tmp);

            llvm::Value *InitialOffsetIdx[] = {
                llvm::ConstantInt::get(CGM.SizeTy, 0), Offset};
            llvm::Value *SharedStructPtr =
                Bld.CreateGEP(SharedData, InitialOffsetIdx);
            return Bld.CreateBitCast(SharedStructPtr, LevelTyPtr);
          };

          // Clone the VLA size loads to before all the uses because the
          // the codegeneration scheme exposes dominance issues.
          for (auto L : VLASizeLoads) {
            for (auto I = L->user_begin(), E = L->user_end(); I != E;) {
              llvm::Instruction *Inst = cast<llvm::Instruction>(*I);
              ++I;

              llvm::Instruction *NewLoad = L->clone();
              NewLoad->insertBefore(Inst);
              Inst->replaceUsesOfWith(L, NewLoad);
            }
            L->eraseFromParent();
          }

          // Now that we have all the storage ready we can replace all the uses
          // of Alloca instructions to addresses in the storage we have just
          // created

          unsigned StructFieldIdx = 0;
          for (unsigned i = 0; i < FLAlloca.size(); ++i) {
            llvm::AllocaInst *AI = FLAlloca[i];
            Bld.SetInsertPoint(AI);

            // If we need to do a dynamic alloc, we need to compute the right
            // size and use malloc.
            if (SharedStackDynamicAlloc) {
              llvm::Value *SelfAddr = Bld.CreateStructGEP(
                  LevelTy, GetSharedStructPtr(false), StructFieldIdx);
              llvm::Value *MallocSize = llvm::ConstantInt::get(
                  CGM.SizeTy, CGM.getModule().getDataLayout().getTypeAllocSize(
                                  AI->getAllocatedType()));

              // multiply by the array size if needed
              if (llvm::ConstantInt *C = FLAllocaSizes[i])
                MallocSize = Bld.CreateMul(
                    MallocSize, Bld.CreateIntCast(C, CGM.SizeTy, false));

              llvm::Value *MallocAddr =
                  Bld.CreateCall(Get_malloc(), MallocSize);
              MallocAddr = Bld.CreateBitCast(MallocAddr, AI->getType());
              Bld.CreateStore(MallocAddr, SelfAddr);

              // For each use of the address we need to load the content in
              // the struct
              for (auto I = AI->user_begin(), E = AI->user_end(); I != E;) {
                llvm::Instruction *Inst = cast<llvm::Instruction>(*I);
                ++I;
                Bld.SetInsertPoint(Inst);
                llvm::Value *Addr = Bld.CreateStructGEP(
                    LevelTy, GetSharedStructPtr(true), StructFieldIdx);
                llvm::Value *LocalAddr = Bld.CreateLoad(AI->getType(), Addr);

                llvm::PointerType *Ty =
                    cast<llvm::PointerType>(LocalAddr->getType());
                llvm::Type *FixedTy =
                    llvm::PointerType::get(Ty->getElementType(), 0);
                LocalAddr = Bld.CreateAddrSpaceCast(LocalAddr, FixedTy);
                Inst->replaceUsesOfWith(AI, LocalAddr);
              }
            } else {
              for (auto I = AI->user_begin(), E = AI->user_end(); I != E;) {
                llvm::Instruction *Inst = cast<llvm::Instruction>(*I);
                ++I;
                Bld.SetInsertPoint(Inst);
                llvm::Value *Addr = Bld.CreateStructGEP(
                    LevelTy, GetSharedStructPtr(true), StructFieldIdx);

                // If this is an array we also need to index the first element
                // of
                // the array
                if (FLAllocaSizes[i])
                  Addr = Bld.CreateConstGEP2_32(AI->getType()->getElementType(),
                                                Addr, 0, 0);

                llvm::PointerType *Ty =
                    cast<llvm::PointerType>(Addr->getType());
                llvm::Type *FixedTy =
                    llvm::PointerType::get(Ty->getElementType(), 0);
                Addr = Bld.CreateAddrSpaceCast(Addr, FixedTy);
                Inst->replaceUsesOfWith(AI, Addr);
              }
            }
            AI->eraseFromParent();
            StructFieldIdx++;
          }

          for (unsigned i = 0; i < VLAlloca.size(); ++i) {
            llvm_unreachable(
                "Variable array types are not currently supported!");
            llvm::AllocaInst *AI = VLAlloca[i];

            Bld.SetInsertPoint(AI);

            assert(VLAllocaSizes[i] &&
                   "Expecting only arrays with a given size!");

            // We need to get the pointer to the actual data, store it in the
            // struct and increment the stack pointer

            llvm::Value *CurrentOffset = Bld.CreateLoad(CGM.SizeTy, SP);

            llvm::Value *DataIndexes[] = {llvm::ConstantInt::get(CGM.SizeTy, 0),
                                          CurrentOffset};
            llvm::Value *DataAddr = Bld.CreateGEP(SharedData, DataIndexes);

            // Cast the pointer to the right type and address space
            llvm::PointerType *DataAddrTy =
                cast<llvm::PointerType>(DataAddr->getType());
            DataAddr = Bld.CreateBitCast(DataAddr,
                                         AI->getAllocatedType()->getPointerTo(
                                             DataAddrTy->getAddressSpace()));
            DataAddr = Bld.CreateAddrSpaceCast(DataAddr, AI->getType());

            llvm::Value *Addr = Bld.CreateStructGEP(
                LevelTy, GetSharedStructPtr(false), StructFieldIdx);
            Bld.CreateStore(DataAddr, Addr);

            CurrentOffset = Bld.CreateAdd(
                CurrentOffset,
                Bld.CreateIntCast(VLAllocaSizes[i], CGM.SizeTy, false));
            Bld.CreateStore(CurrentOffset, SP);

            // For each use of the address we need to load the content in
            // the struct
            for (auto I = AI->user_begin(), E = AI->user_end(); I != E;) {
              llvm::Instruction *Inst = cast<llvm::Instruction>(*I);
              ++I;
              Bld.SetInsertPoint(Inst);
              llvm::Value *Addr = Bld.CreateStructGEP(
                  LevelTy, GetSharedStructPtr(true), StructFieldIdx);
              llvm::Value *LocalAddr = Bld.CreateLoad(AI->getType(), Addr);

              llvm::PointerType *Ty =
                  cast<llvm::PointerType>(LocalAddr->getType());
              llvm::Type *FixedTy =
                  llvm::PointerType::get(Ty->getElementType(), 0);
              LocalAddr = Bld.CreateAddrSpaceCast(LocalAddr, FixedTy);
              Inst->replaceUsesOfWith(AI, LocalAddr);
            }

            AI->eraseFromParent();
            StructFieldIdx++;
          }
        }
      }
    }

    // Make sure the static entries are turned visible
    for (auto G : StaticEntries) {
      std::string NewName = "__omptgt__static_";
      NewName += CGM.getLangOpts().OMPModuleUniqueID;
      NewName += "__";
      NewName += G->getName();
      G->setName(NewName);
      G->setLinkage(llvm::GlobalValue::ExternalLinkage);
    }

    // StackSave/Restore seem to not be currently supported by the backend
    if (CGM.getModule().getFunction("llvm.stacksave")) {
      llvm_unreachable("Variable array types are not currently supported!");
    }
    if (CGM.getModule().getFunction("llvm.stackrestore")) {
      llvm_unreachable("Variable array types are not currently supported!");
    }

    // Legalize names of globals and functions.
    // FIXME: This should be moved to the backend.
    for (auto &I : CGM.getModule().getGlobalList()) {
      if (!I.hasInternalLinkage())
        continue;
      if (I.getName().find('.') == I.getName().npos)
        continue;

      std::string N = I.getName();
      std::replace(N.begin(), N.end(), '.', '_');
      I.setName(Twine("__ptxnamefix__") + Twine(N));
    }
    for (auto &I : CGM.getModule().getFunctionList()) {
      if (I.isIntrinsic())
        continue;
      if (I.getName().find('.') == I.getName().npos)
        continue;

      std::string N = I.getName();
      std::replace(N.begin(), N.end(), '.', '_');
      I.setName(Twine("__ptxnamefix__") + Twine(N));
    }

    CGOpenMPRuntime::PostProcessModule(CGM);

    // Process printf calls
    PostProcessPrintfs(CGM.getModule());
  }

  void registerCtorRegion(llvm::Function *Fn) {
    assert(CGM.getLangOpts().OpenMPTargetMode);
    std::string Name = Fn->getName();

    // Add dummy global for thread_limit
    new llvm::GlobalVariable(CGM.getModule(), CGM.Int32Ty, true,
                             llvm::GlobalValue::ExternalLinkage,
                             llvm::Constant::getNullValue(CGM.Int32Ty),
                             Name + Twine("_thread_limit"));

    CGOpenMPRuntime::registerCtorRegion(Fn);
  }
  void registerDtorRegion(llvm::Function *Fn, llvm::Constant *Destructee) {
    assert(CGM.getLangOpts().OpenMPTargetMode);
    std::string Name = Fn->getName();

    // Add dummy global for thread_limit
    new llvm::GlobalVariable(CGM.getModule(), CGM.Int32Ty, true,
                             llvm::GlobalValue::ExternalLinkage,
                             llvm::Constant::getNullValue(CGM.Int32Ty),
                             Name + Twine("_thread_limit"));

    CGOpenMPRuntime::registerDtorRegion(Fn, Destructee);

    return;
  }

private:
  llvm::Value *GetTeamReduFuncGeneral(CodeGenFunction &CGF, QualType QTyRes,
                                      QualType QTyIn,
                                      CGOpenMPRuntime::EAtomicOperation Aop) {
    SmallString<40> Str;
    llvm::raw_svector_ostream OS(Str);

    if (QTyRes.isVolatileQualified() || QTyIn.isVolatileQualified())
      return 0;

    int64_t TySize =
        CGF.CGM.GetTargetTypeStoreSize(CGF.ConvertTypeForMem(QTyRes))
            .getQuantity();
    if (QTyRes->isRealFloatingType()) {
      OS << "__gpu_warpBlockRedu_float";
      if (TySize != 4 && TySize != 8 && TySize != 10 && TySize != 16)
        return 0;
    } else if (QTyRes->isComplexType()) {
      OS << "__gpu_warpBlockRedu_cmplx";
      if (TySize != 8 && TySize != 16)
        return 0;
    } else if (QTyRes->isScalarType()) {
      OS << "__gpu_warpBlockRedu_fixed";
      if (TySize != 1 && TySize != 2 && TySize != 4 && TySize != 8)
        return 0;
    } else
      return 0;
    if (QTyRes->isComplexType()) {
      OS << TySize / 2;
    } else {
      OS << TySize;
    }
    switch (Aop) {
    case OMP_Atomic_orl:
      OS << "_orl";
      break;
    case OMP_Atomic_orb:
      OS << "_orb";
      break;
    case OMP_Atomic_andl:
      OS << "_andl";
      break;
    case OMP_Atomic_andb:
      OS << "_andb";
      break;
    case OMP_Atomic_xor:
      OS << "_xor";
      break;
    case OMP_Atomic_sub:
      OS << "_sub";
      break;
    case OMP_Atomic_add:
      OS << "_add";
      break;
    case OMP_Atomic_mul:
      OS << "_mul";
      break;
    case OMP_Atomic_div:
      if (QTyRes->hasUnsignedIntegerRepresentation() ||
          QTyRes->isPointerType()) {
        if (!CGF.getContext().hasSameType(QTyIn, QTyRes))
          return 0;
        OS << "u";
      }
      OS << "_div";
      break;
    case OMP_Atomic_min:
      OS << "_min";
      break;
    case OMP_Atomic_max:
      OS << "_max";
      break;
    case OMP_Atomic_shl:
      OS << "_shl";
      break;
    case OMP_Atomic_shr:
      if (QTyRes->hasUnsignedIntegerRepresentation() ||
          QTyRes->isPointerType()) {
        if (!CGF.getContext().hasSameType(QTyIn, QTyRes))
          return 0;
        OS << "u";
      }
      OS << "_shr";
      break;
    case OMP_Atomic_wr:
      OS << "_wr";
      break;
    case OMP_Atomic_rd:
      OS << "_rd";
      break;
    case OMP_Atomic_assign:
      return 0;
    case OMP_Atomic_invalid:
    default:
      llvm_unreachable("Unknown atomic operation.");
    }
    int64_t TyInSize =
        CGF.CGM.GetTargetTypeStoreSize(CGF.ConvertTypeForMem(QTyIn))
            .getQuantity();
    if (!CGF.getContext().hasSameType(QTyIn, QTyRes)) {
      if (QTyRes->isScalarType() && QTyIn->isRealFloatingType() &&
          TyInSize == 8)
        OS << "_float8";
      else
        return 0;
    }
    SmallVector<llvm::Type *, 1> Params;
    llvm::Type *Ty = CGF.ConvertTypeForMem(GetAtomicType(CGF, QTyRes));
    Params.push_back(Ty);
    llvm::Type *RetTy = Ty;
    llvm::FunctionType *FunTy = llvm::FunctionType::get(RetTy, Params, false);
    return CGF.CGM.CreateRuntimeFunction(FunTy, OS.str());
  }

public:
  llvm::Value *GetTeamReduFunc(CodeGenFunction &CGF, QualType QTy,
                               OpenMPReductionClauseOperator Op) {

    if (QTy.isVolatileQualified())
      return 0;

    EAtomicOperation Aop = OMP_Atomic_invalid;
    switch (Op) {
    case OMPC_REDUCTION_or:
      Aop = OMP_Atomic_orl;
      break;
    case OMPC_REDUCTION_bitor:
      Aop = OMP_Atomic_orb;
      break;
    case OMPC_REDUCTION_and:
      Aop = OMP_Atomic_andl;
      break;
    case OMPC_REDUCTION_bitand:
      Aop = OMP_Atomic_andb;
      break;
    case OMPC_REDUCTION_bitxor:
      Aop = OMP_Atomic_xor;
      break;
    case OMPC_REDUCTION_sub:
      Aop = OMP_Atomic_add;
      break;
    case OMPC_REDUCTION_add:
      Aop = OMP_Atomic_add;
      break;
    case OMPC_REDUCTION_mult:
      Aop = OMP_Atomic_mul;
      break;
    case OMPC_REDUCTION_min:
      Aop = OMP_Atomic_min;
      break;
    case OMPC_REDUCTION_max:
      Aop = OMP_Atomic_max;
      break;
    case OMPC_REDUCTION_custom:
      return 0;
    case OMPC_REDUCTION_unknown:
    case NUM_OPENMP_REDUCTION_OPERATORS:
      llvm_unreachable("Unknown reduction operation.");
    }
    return GetTeamReduFuncGeneral(CGF, QTy, QTy, Aop);
  }

  llvm::Value * Get_kmpc_print_int() {
    return CGM.CreateRuntimeFunction(
       llvm::TypeBuilder<___kmpc_print_int, false>::get(
           CGM.getLLVMContext()), "__kmpc_print_int");
  }
  llvm::Value * Get_kmpc_print_address_int64() {
    return CGM.CreateRuntimeFunction(
       llvm::TypeBuilder<__kmpc_print_address_int64, false>::get(
           CGM.getLLVMContext()), "__kmpc_print_address_int64");
  }
}; // class CGOpenMPRuntime_NVPTX

///===---------------
///
/// Create runtime for the target used in the Module
///
///===---------------

CGOpenMPRuntime *CodeGen::CreateOpenMPRuntime(CodeGenModule &CGM) {

  switch (CGM.getTarget().getTriple().getArch()) {
  default:
    return new CGOpenMPRuntime(CGM);
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return new CGOpenMPRuntime_NVPTX(CGM);
  }

}
