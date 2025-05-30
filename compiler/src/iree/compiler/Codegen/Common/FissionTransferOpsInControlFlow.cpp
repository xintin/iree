// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fission-transfer-ops-in-control-flow"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FISSIONTRANSFEROPSINCONTROLFLOWPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Note: this should exists in mlir/lib/Dialect/IR/GPUDialect.cpp
bool isPrivateAddressSpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == gpu::GPUDialect::getPrivateAddressSpace();
  return false;
}

void replaceIterationVariable(Operation *op, Value iterArg, Value constant) {
  for (auto &operand : op->getOpOperands()) {
    if (operand.get() == iterArg) {
      operand.set(constant);
    }
  }
}

SetVector<Operation *> collectBackwardSliceInControlFlow(Operation *op,
                                                         Operation *parentOp) {
  BackwardSliceOptions options;
  options.inclusive = false;
  options.filter = [&](Operation *op) { return parentOp == op->getParentOp(); };
  SetVector<Operation *> slice;
  getBackwardSlice(op, &slice, options);
  return slice;
}

void cloneSliceIntoLoop(IRRewriter &rewriter, SetVector<Operation *> &slice,
                        scf::ForOp &newLoop, IRMapping &mapping) {
  for (Operation *op : slice) {
    rewriter.clone(*op, mapping);
  }
}

scf::ForOp createNewLoop(IRRewriter &rewriter, scf::ForOp forOp, Location loc) {
  return rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep(),
                                     forOp.getRegionIterArgs());
}

memref::AllocaOp createAlloca(IRRewriter &rewriter,
                              vector::TransferWriteOp writeOp) {
  auto allocaType = cast<MemRefType>(writeOp.getBase().getType());
  auto allocaTypeNoStride =
      MemRefType::Builder(allocaType.getShape(), allocaType.getElementType());
  return rewriter.create<memref::AllocaOp>(writeOp.getLoc(),
                                           allocaTypeNoStride);
}

/// Splits transfer read and write operations from a control flow Operation
/// (forOp) into separate loops.
///
/// For example, given a loop with transfer read and write operations:
///   scf.for %i = 0 to 10 {
///     %read = vector.transfer_read ...
///     vector.transfer_write %read ...
///   }
///
/// This function will transform it into:
///   %alloca = memref.alloca ...  // Alloca for intermediate results
///   scf.for %i = 0 to 10 {
///     %read = vector.transfer_read ...
///     vector.transfer_write %read %alloca
///   }
///   scf.for %j = 0 to 10 {
///     %read = vector.transfer_read %alloca
///     vector.transfer_write %read ...
///   }
void splitTransferOpsFromControlFlow(IRRewriter &rewriter,
                                     vector::TransferReadOp readOp,
                                     vector::TransferWriteOp writeOp,
                                     scf::ForOp forOp) {
  DBGS() << "Splitting transfer ops from control flow: \n"
         << "For Op: " << forOp << "\n";

  rewriter.setInsertionPoint(forOp);
  memref::AllocaOp alloca = createAlloca(rewriter, writeOp);

  // Read loop
  scf::ForOp readLoop = createNewLoop(rewriter, forOp, readOp.getLoc());
  rewriter.setInsertionPointToStart(readLoop.getBody());

  IRMapping readMapping;
  readMapping.map(forOp.getInductionVar(), readLoop.getInductionVar());
  SetVector<Operation *> readSlice =
      collectBackwardSliceInControlFlow(readOp, forOp);
  cloneSliceIntoLoop(rewriter, readSlice, readLoop, readMapping);

  Operation *lastRead = rewriter.clone(*readOp, readMapping);
  auto newTransferWrite = writeOp.clone();
  newTransferWrite->setOperand(0, lastRead->getResult(0));
  newTransferWrite->setOperand(1, alloca);
  rewriter.insert(newTransferWrite);

  DBGS() << "Read loop: \n" << readLoop << "\n";

  // Write loop
  rewriter.setInsertionPointAfter(readLoop.getOperation());
  scf::ForOp writeLoop = createNewLoop(rewriter, forOp, writeOp.getLoc());
  rewriter.setInsertionPointToStart(writeLoop.getBody());

  vector::TransferReadOp newReadOp = readOp.clone();
  newReadOp->setOperand(0, alloca);
  rewriter.insert(newReadOp);

  IRMapping writeMapping;
  writeMapping.map(forOp.getInductionVar(), writeLoop.getInductionVar());
  SetVector<Operation *> writeSlice =
      collectBackwardSliceInControlFlow(writeOp, forOp);
  cloneSliceIntoLoop(rewriter, writeSlice, writeLoop, writeMapping);

  auto lastWrite = rewriter.clone(*writeOp, writeMapping);
  lastWrite->setOperand(0, newReadOp->getResult(0));

  for (Operation &op : writeLoop.getBody()->getOperations()) {
    for (auto &operand : op.getOpOperands()) {
      if (operand.get() == writeMapping.lookup(readOp.getResult())) {
        operand.set(newReadOp);
      }
    }
  }
  DBGS() << "Write loop: \n" << writeLoop << "\n";

  rewriter.eraseOp(forOp);
}

struct FissionTarget {
  Operation *parent;
  vector::TransferReadOp readOp;
  vector::TransferWriteOp writeOp;
};

static FailureOr<FissionTarget> processReadOp(vector::TransferReadOp readOp) {
  auto parentOp = readOp->getParentOp();
  // auto parentOp = forOp;
  if (!parentOp || !isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parentOp)) {
    return failure();
  }

  auto base = readOp.getBase();
  auto addrspace = cast<MemRefType>(base.getType()).getMemorySpace();
  if (gpu::GPUDialect::isWorkgroupMemoryAddressSpace(addrspace) ||
      isPrivateAddressSpace(addrspace)) {
    return failure();
  }

  ForwardSliceOptions options;
  options.inclusive = false;
  options.filter = [&](Operation *op) { return parentOp == op->getParentOp(); };
  SetVector<Operation *> slice;
  getForwardSlice(readOp.getOperation(), &slice, options);

  bool hasWriteOp = false;
  FissionTarget fissionTarget;
  for (Operation *op : slice) {
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      auto writeBase = writeOp.getBase();
      auto writeAddrspace =
          cast<MemRefType>(writeBase.getType()).getMemorySpace();
      // if (isPrivateAddressSpace(writeAddrspace)) {
      if (isPrivateAddressSpace(writeAddrspace) ||
          gpu::GPUDialect::isWorkgroupMemoryAddressSpace(writeAddrspace)) {
        scf::ForOp parentForOp = cast<scf::ForOp>(parentOp);
        if (writeOp != parentForOp.getBody()->getTerminator()->getPrevNode()) {
          continue;
        }
        // Only consider transfer_write ops that are in the same address space.
        fissionTarget = {parentOp, readOp, writeOp};
        hasWriteOp = true;
        // DBGS() << "Fissioning target readOp: " << readOp
        //        << " and writeOp: " << writeOp
        //        //<< " in parent operation: " <<
        //        cast<scf::ForOp>(target.parent)
        //        << "\n";
      }
    }
  }
  if (!hasWriteOp) {
    return failure();
  }
  return fissionTarget;
}

static FailureOr<SmallVector<FissionTarget>>
populateFissionTargets(scf::ForOp forOp) {

  SmallVector<FissionTarget> fissionTargets;
  forOp->walk([&](Operation *op) {
    if (op->getParentOp() != forOp) {
      return;
    }

    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      auto result = processReadOp(readOp);
      if (failed(result)) {
        return;
      }
      fissionTargets.push_back(result.value());
      for (const FissionTarget &target : fissionTargets) {
        DBGS() << "Fissioning target readOp: " << target.readOp
               << " and writeOp: "
               << target.writeOp
               //       << " in parent operation: " <<
               //       cast<scf::ForOp>(target.parent)
               << "\n";
      }
    }
  });
  return fissionTargets;
}

struct FissionTransferOpsInControlFlowPass final
    : impl::FissionTransferOpsInControlFlowPassBase<
          FissionTransferOpsInControlFlowPass> {
public:
  void runOnOperation() override;
};

} // namespace

void FissionTransferOpsInControlFlowPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());

  SmallVector<scf::ForOp> loops;
  funcOp.walk([&loops](scf::ForOp forOp) { loops.push_back(forOp); });

  SmallVector<FissionTarget> fissionTargets;
  for (scf::ForOp forOp : loops) {
    auto result = populateFissionTargets(forOp);
    if (failed(result)) {
      continue;
    }
    fissionTargets.insert(fissionTargets.end(), result.value().begin(),
                          result.value().end());
  }

  for (const FissionTarget &target : fissionTargets) {
    if (isa<scf::ForOp>(target.parent)) {
      splitTransferOpsFromControlFlow(rewriter, target.readOp, target.writeOp,
                                      cast<scf::ForOp>(target.parent));
    }
  }
}

} // namespace mlir::iree_compiler
