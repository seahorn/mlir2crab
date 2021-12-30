//===- CrabOps.cpp - Crab dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir2crab/Dialect/Crab/IR/CrabDialect.h"
#include "mlir2crab/Dialect/Crab/IR/CrabOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::crab;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter &p, mlir::crab::ConstantOp &op) {
  p << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  p << op.getValue();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(valueAttr.getType(), result.types);
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

#define GET_OP_CLASSES
#include "Dialect/Crab/IR/CrabOps.cpp.inc"
