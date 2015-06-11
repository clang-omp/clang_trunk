//===---- CGLoopInfo.cpp - LLVM CodeGen for loop metadata -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CGLoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
using namespace clang::CodeGen;
using namespace llvm;

static llvm::MDNode *CreateMetadata(llvm::LLVMContext &Ctx,
                                    const LoopAttributes &Attrs) {
  using namespace llvm;

  if (!Attrs.IsParallel &&
      Attrs.VectorizerWidth == 0 &&
      Attrs.VectorizerEnable == LoopAttributes::LVEC_UNSPECIFIED)
    return 0;

  SmallVector<Metadata *, 4> Args;
  // Reserve operand 0 for loop id self reference.
  auto TempNode = MDNode::getTemporary(Ctx, None);
  Args.push_back(TempNode.get());

  // Setting vectorizer.width
  // TODO: For a correct implementation of 'safelen' clause
  // we need to update the value somewhere (based on target info).
  if (Attrs.VectorizerWidth > 0) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.vectorize.width"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), Attrs.VectorizerWidth))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

#if 0
  // Setting vectorizer.unroll
  if (Attrs.VectorizerUnroll > 0) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.interleave.count"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), Attrs.VectorizerUnroll))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }
#endif

  // Setting vectorizer.enable
  if (Attrs.VectorizerEnable != LoopAttributes::LVEC_UNSPECIFIED) {
    Metadata *Vals[] = {
        MDString::get(Ctx, "llvm.loop.vectorize.enable"),
        ConstantAsMetadata::get(ConstantInt::get(
            Type::getInt1Ty(Ctx),
            (Attrs.VectorizerEnable == LoopAttributes::LVEC_ENABLE)))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Set the first operand to itself.
  MDNode *LoopID = MDNode::get(Ctx, Args);
  LoopID->replaceOperandWith(0, LoopID);
  return LoopID;
}

LoopAttributes::LoopAttributes(bool IsParallel)
  : IsParallel(IsParallel),
    VectorizerEnable(LoopAttributes::LVEC_UNSPECIFIED),
    VectorizerWidth(0) { }

void LoopAttributes::Clear() {
  IsParallel = false;
  VectorizerWidth = 0;
  VectorizerEnable = LoopAttributes::LVEC_UNSPECIFIED;
}

LoopInfo::LoopInfo(llvm::BasicBlock *Header, const LoopAttributes &Attrs)
  : LoopID(0), Header(Header), Attrs(Attrs) {
  LoopID = CreateMetadata(Header->getContext(), Attrs);
}

LoopInfo::LoopInfo(llvm::MDNode *LoopID, const LoopAttributes &Attrs)
  : LoopID(LoopID), Header(0), Attrs(Attrs) { }

void LoopInfoStack::Push(llvm::BasicBlock *Header) {
  Active.push_back(LoopInfo(Header, StagedAttrs));
  // Clear the attributes so nested loops do not inherit them.
  StagedAttrs.Clear();
}

void LoopInfoStack::Pop() {
  assert(!Active.empty());
  Active.pop_back();
}

void LoopInfoStack::AddAligned(const llvm::Value *Val, int Align) {
  // The following restriction should be enforced by Sema, so
  // check it with assertion.
  assert(Aligneds.find(Val) == Aligneds.end() ||
         Aligneds.find(Val)->second == Align);
  Aligneds.insert(std::make_pair(Val, Align));
}

int LoopInfoStack::GetAligned(const llvm::Value *Val) const {
  llvm::DenseMap<const llvm::Value *, int>::const_iterator It =
    Aligneds.find(Val);
  if (It == Aligneds.end()) return 0;
  return It->second;
}

void LoopInfoStack::InsertHelper(llvm::Instruction *I) const {
  if (!HasInfo())
    return;

  const LoopInfo &L = GetInfo();

  if (!L.GetLoopID())
    return;

  if (llvm::TerminatorInst *TI = llvm::dyn_cast<llvm::TerminatorInst>(I)) {
    for (unsigned i = 0, ie = TI->getNumSuccessors(); i < ie; ++i)
      if (TI->getSuccessor(i) == L.GetHeader()) {
        TI->setMetadata("llvm.loop", L.GetLoopID());
        break;
      }
    return;
  }

  if (L.GetAttributes().IsParallel) {
    if (llvm::StoreInst *SI = llvm::dyn_cast<llvm::StoreInst>(I)) {
      SI->setMetadata("llvm.mem.parallel_loop_access", L.GetLoopID());
    }
    else if (llvm::LoadInst *LI = llvm::dyn_cast<llvm::LoadInst>(I)) {
      LI->setMetadata("llvm.mem.parallel_loop_access", L.GetLoopID());
    }
  }
}

void LoopInfoStack::Push(llvm::MDNode *LoopID, bool IsParallel) {
  assert(Active.empty() && "cannot have an active loop");
  Active.push_back(LoopInfo(LoopID, LoopAttributes(IsParallel)));
  StagedAttrs.Clear();
}

