#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <typename T> static T convertType(T type) { return type; }

template <>
RankedTensorType convertType<RankedTensorType>(RankedTensorType type) {
  auto encoding = type.getEncoding();
  auto blockedEncoding = dyn_cast<ttg::BlockedEncodingAttr>(encoding);
  if (!blockedEncoding)
    return type;
  // fixme: set sizePerWarp directly instead of this work-around
  auto sizePerThread = blockedEncoding.getSizePerThread();
  auto threadsPerWarp = blockedEncoding.getThreadsPerWarp();
  llvm::SmallVector<long> sizePerWarp;
  for (auto [lhs, rhs] : llvm::zip(sizePerThread, threadsPerWarp)) {
    sizePerWarp.push_back(lhs * rhs);
  }
  auto newType =
      RankedTensorType::get(sizePerWarp, type.getElementType(), encoding);
  return newType;
}

template <> tt::PointerType convertType<tt::PointerType>(tt::PointerType type) {
  auto pointeeType = type.getPointeeType();
  auto tensorType = dyn_cast<RankedTensorType>(pointeeType);
  if (!tensorType)
    return type;
  auto newTensorType = convertType(tensorType);
  auto newType = tt::PointerType::get(newTensorType, type.getAddressSpace());
  return newType;
}

void dispatchGenericOp(Operation *op) {
  OpBuilder b(op);
  auto newOp = b.clone(*op);
  for (auto result : newOp->getResults()) {
    if (auto castType = dyn_cast<RankedTensorType>(result.getType()))
      result.setType(convertType(castType));
    else if (auto castType = dyn_cast<tt::PointerType>(result.getType()))
      result.setType(convertType(castType));
  }
  newOp->dump();
  op->replaceAllUsesWith(newOp->getResults());
  op->erase();
  return;
}

void dispatchArithConstantOp(arith::ConstantOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  if (!type)
    return;
  if (!isa<ttg::BlockedEncodingAttr>(type.getEncoding()))
    return;
  auto newType = convertType(type);
  auto value = cast<DenseElementsAttr>(op.getValue());
  value = value.resizeSplat(newType);
  OpBuilder b(op);
  auto newOp = b.create<arith::ConstantOp>(op.getLoc(), newType, value);
  addNamedAttrs(newOp, op->getAttrDictionary());
  newOp.dump();
  op->replaceAllUsesWith(newOp->getResults());
  op->erase();
  return;
}

void dispatchMakeTensorPtrOp(tt::MakeTensorPtrOp op, Value warpId) {
  auto loc = op.getLoc();
  auto type = op.getType();
  auto newType = convertType(type);
  OpBuilder b(op);
  auto offsets = op.getOffsets();
  SmallVector<Value> newOffsets;
  // fixme: make it naive for now
  for (auto offset : offsets) {
    auto newOffset = b.create<arith::AddIOp>(loc, warpId, offset);
    newOffsets.push_back(newOffset);
  }
  auto newOp = b.clone(*op.getOperation());
  auto newPtrOp = cast<tt::MakeTensorPtrOp>(newOp);
  newPtrOp.getOffsetsMutable().assign(newOffsets);
  newPtrOp.getResult().setType(newType);
  newPtrOp.dump();
  op->replaceAllUsesWith(newPtrOp->getResults());
  op->erase();
  return;
}

void dispatchScfForOp(scf::ForOp op) {
  auto body = op.getBody();
  for (auto [lhs, rhs] :
       llvm::zip(body->getArguments().drop_front(1), op.getInitArgs()))
    lhs.setType(rhs.getType());
  for (auto result : op->getResults()) {
    if (auto castType = dyn_cast<RankedTensorType>(result.getType()))
      result.setType(convertType(castType));
    else if (auto castType = dyn_cast<tt::PointerType>(result.getType()))
      result.setType(convertType(castType));
  }
  op.dump();
  return;
}

// fixme : TBD
// void dispatchConvertLayoutOp(ttg::ConvertLayoutOp op) {
// i,j from source layout; m,n from dest layout
// %st_addr = tt.make_tensor_ptr xxx, ..., : <tensor<ixjxf16, #blockedA>, 3>
// tt.store %st_addr, %A {...} : tensor<ixjxf16>
// triton_gpu.sync_threads
// %ld_addr = tt.make_tensor_ptr xxx, ..., : <tensor<mxnxf16, #blockedA>, 3>
// %ld = tt.load %ld_addr : tensor<mxnxf16>
// }

// void dispatchScfForOp(scf::ForOp op) {
//   OpBuilder b(op);
//   auto scf =
//       b.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
//       op.getUpperBound(),
//                            op.getStep(), op.getInitArgs());
//   // merge block
//   auto body = op.getBody();
//   auto scfBody = scf.getBody();
//   for (auto [lhs, rhs] :
//        llvm::zip(body->getArguments(), scfBody->getArguments()))
//     lhs.replaceAllUsesWith(rhs);
//   scfBody->getOperations().splice(scfBody->end(), body->getOperations());
//   scf.dump();
//   op.replaceAllUsesWith(scf.getResults());
//   op.erase();
//   return;
// }

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUDispatchToWarpsPass
    : public TritonGPUDispatchToWarpsBase<TritonGPUDispatchToWarpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    for (auto func : m.getOps<tt::FuncOp>()) {
      auto b = OpBuilder::atBlockBegin(&func.getBody().front());
      auto warpId = b.create<tt::GetWarpIdOp>(func.getLoc());
      // bool changed = false;
      // func.walk([&](Operation *op) {
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (llvm::all_of(op->getResultTypes(), [&](Type type) {
              return !isa<RankedTensorType>(type) &&
                     !isa<tt::PointerType>(type);
            }))
          ;
        else if (auto forOp = dyn_cast<scf::ForOp>(op))
          dispatchScfForOp(forOp);
        else if (auto ptrOp = dyn_cast<tt::MakeTensorPtrOp>(op))
          dispatchMakeTensorPtrOp(ptrOp, warpId);
        else if (auto cst = dyn_cast<arith::ConstantOp>(op))
          dispatchArithConstantOp(cst);
        else if (isa<tt::LoadOp, tt::DotOp, tt::AdvanceOp, ttg::ConvertLayoutOp,
                     arith::TruncFOp>(op))
          dispatchGenericOp(op);
        else
          assert(0 && "op not considered");
        return WalkResult::advance();
      });
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUDispatchToWarpsPass() {
  return std::make_unique<TritonGPUDispatchToWarpsPass>();
}
