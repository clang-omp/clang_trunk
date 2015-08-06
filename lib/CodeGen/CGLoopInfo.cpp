//===---- CGLoopInfo.cpp - LLVM CodeGen for loop metadata -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CGLoopInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/LoopHint.h"
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

  if (!Attrs.IsParallel && Attrs.VectorizeWidth == 0 &&
      Attrs.InterleaveCount == 0 && Attrs.UnrollCount == 0 &&
      Attrs.VectorizeEnable == LoopAttributes::Unspecified &&
      Attrs.UnrollEnable == LoopAttributes::Unspecified)
    return nullptr;

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

  // Setting vectorizer.unroll
  if (Attrs.VectorizerUnroll > 0) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.interleave.count"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), Attrs.VectorizerUnroll))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Setting interleave.count
  if (Attrs.UnrollCount > 0) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.unroll.count"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), Attrs.UnrollCount))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Setting vectorize.enable
  if (Attrs.VectorizeEnable != LoopAttributes::Unspecified) {
    Metadata *Vals[] = {MDString::get(Ctx, "llvm.loop.vectorize.enable"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt1Ty(Ctx), (Attrs.VectorizeEnable ==
                                                   LoopAttributes::Enable)))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Setting unroll.full or unroll.disable
  if (Attrs.UnrollEnable != LoopAttributes::Unspecified) {
    Metadata *Vals[] = {
        MDString::get(Ctx, (Attrs.UnrollEnable == LoopAttributes::Enable
                                ? "llvm.loop.unroll.full"
                                : "llvm.loop.unroll.disable"))};
    Args.push_back(MDNode::get(Ctx, Vals));
  }

  // Set the first operand to itself.
  MDNode *LoopID = MDNode::get(Ctx, Args);
  LoopID->replaceOperandWith(0, LoopID);
  return LoopID;
}

LoopAttributes::LoopAttributes(bool IsParallel)
    : IsParallel(IsParallel), VectorizeEnable(LoopAttributes::Unspecified),
      UnrollEnable(LoopAttributes::Unspecified), VectorizeWidth(0),
      InterleaveCount(0), UnrollCount(0) {}

void LoopAttributes::Clear() {
  IsParallel = false;
  VectorizeWidth = 0;
  InterleaveCount = 0;
  UnrollCount = 0;
  VectorizeEnable = LoopAttributes::Unspecified;
  UnrollEnable = LoopAttributes::Unspecified;
}

LoopInfo::LoopInfo(llvm::BasicBlock *Header, const LoopAttributes &Attrs)
  : LoopID(0), Header(Header), Attrs(Attrs) {
  LoopID = CreateMetadata(Header->getContext(), Attrs);
}

void LoopInfoStack::push(BasicBlock *Header) {
  Active.push_back(LoopInfo(Header, StagedAttrs));
  // Clear the attributes so nested loops do not inherit them.
  StagedAttrs.clear();
}

void LoopInfoStack::push(BasicBlock *Header, clang::ASTContext &Ctx,
                         ArrayRef<const clang::Attr *> Attrs) {

  // Identify loop hint attributes from Attrs.
  for (const auto *Attr : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(Attr);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    auto *ValueExpr = LH->getValue();
    unsigned ValueInt = 1;
    if (ValueExpr) {
      llvm::APSInt ValueAPS = ValueExpr->EvaluateKnownConstInt(Ctx);
      ValueInt = ValueAPS.getSExtValue();
    }

    LoopHintAttr::OptionType Option = LH->getOption();
    LoopHintAttr::LoopHintState State = LH->getState();
    switch (State) {
    case LoopHintAttr::Disable:
      switch (Option) {
      case LoopHintAttr::Vectorize:
        // Disable vectorization by specifying a width of 1.
        setVectorizeWidth(1);
        break;
      case LoopHintAttr::Interleave:
        // Disable interleaving by speciyfing a count of 1.
        setInterleaveCount(1);
        break;
      case LoopHintAttr::Unroll:
        setUnrollEnable(false);
        break;
      case LoopHintAttr::UnrollCount:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::InterleaveCount:
        llvm_unreachable("Options cannot be disabled.");
        break;
      }
      break;
    case LoopHintAttr::Enable:
      switch (Option) {
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::Interleave:
        setVectorizeEnable(true);
        break;
      case LoopHintAttr::Unroll:
        setUnrollEnable(true);
        break;
      case LoopHintAttr::UnrollCount:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::InterleaveCount:
        llvm_unreachable("Options cannot enabled.");
        break;
      }
      break;
    case LoopHintAttr::AssumeSafety:
      switch (Option) {
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::Interleave:
        // Apply "llvm.mem.parallel_loop_access" metadata to load/stores.
        SetParallel(true);
      }
      break;
    case LoopHintAttr::Default:
      switch (Option) {
      case LoopHintAttr::VectorizeWidth:
        setVectorizeWidth(ValueInt);
        break;
      case LoopHintAttr::InterleaveCount:
        setInterleaveCount(ValueInt);
        break;
      case LoopHintAttr::UnrollCount:
        setUnrollCount(ValueInt);
        break;
      case LoopHintAttr::Unroll:
        // The default option is used when '#pragma unroll' is specified.
        setUnrollEnable(true);
        break;
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::Interleave:
        llvm_unreachable("Options cannot be assigned a value and do not have a "
                         "default value.");
        break;
      }
      break;
    }
  }

  /// Stage the attributes.
  push(Header);
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

