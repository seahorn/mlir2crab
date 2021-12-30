//===- CrabOps.td - Crab dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRAB_OPS
#define CRAB_OPS

include "CrabDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Crab definition for Booleans
//===----------------------------------------------------------------------===//
def CrabBool: I<1>;

//===----------------------------------------------------------------------===//
// Crab definition for integers
//===----------------------------------------------------------------------===//
// definition of AnySignlessInteger
def CrabInteger: Type<
  CPred<"$_self.isSignlessInteger()">, "signless integer",
        "::mlir::IntegerType">;

//===----------------------------------------------------------------------===//
// Crab binary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for Crab arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class CrabArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Crab_Dialect, mnemonic, traits #[NoSideEffect]
    #ElementwiseMappable.traits> { }

// This operation takes two operands and returns one result,
// each of these is required to be of the same type.
// The custom assembly form of the operation is as follows
//
//     <op> %0, %1 : i32
class CrabIntBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    CrabArithmeticOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins CrabInteger:$lhs, CrabInteger:$rhs)> {

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];
}

def IntAddOp : CrabIntBinaryOp<"add", [Commutative]> {
    let summary = "integer addition operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.add(%lhs, %rhs) : i32
        ```
    }];
}

def IntSubOp : CrabIntBinaryOp<"sub"> {
    let summary = "integer subtraction operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.sub(%lhs, %rhs) : i32
        ```
    }];
}

def IntMulOp : CrabIntBinaryOp<"mul", [Commutative]> {
    let summary = "integer multiplication operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.mul(%lhs, %rhs) : i32
        ```
    }];
}

def SModOp : CrabIntBinaryOp<"smod"> {
    let summary = "integer signed modulus operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.smod(%lhs, %rhs) : i32
        ```
    }];
}

def SRemOp : CrabIntBinaryOp<"srem"> {
    let summary = "integer signed remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.srem(%lhs, %rhs) : i32
        ```
    }];
}

def URemOp : CrabIntBinaryOp<"urem"> {
    let summary = "integer unsigned remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.urem(%lhs, %rhs) : i32
        ```
    }];
}

def SDivOp : CrabIntBinaryOp<"sdiv"> {
    let summary = "integer signed division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.sdiv(%lhs, %rhs) : i32
        ```
    }];
}

def UDivOp : CrabIntBinaryOp<"udiv"> {
    let summary = "integer unsigned division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.udiv(%lhs, %rhs) : i32
        ```
    }];
}

def OrOp : CrabIntBinaryOp<"or", [Commutative]> {
    let summary = "integer binary and operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.or(%lhs, %rhs) : i32
        ```
    }];
}

def AndOp : CrabIntBinaryOp<"and", [Commutative]> {
    let summary = "integer binary and operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.and(%lhs, %rhs) : i32
        ```
    }];
}


def XOrOp : CrabIntBinaryOp<"xor", [Commutative]> {
    let summary = "integer binary xor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.xor(%lhs, %rhs) : i32
        ```
    }];
}

//===----------------------------------------------------------------------===//
// crab constraint ops definitions
//===----------------------------------------------------------------------===//

// Base class for crab constraint operations.  Requires operands to be
// of the same type, but does not constrain them to specific
// types.
//  The custom assembly form of the operation is as follows
//
//     <op> "predicate"(%0, %1)
class CrabConstraintOp<string mnemonic, list<OpTrait> traits = [SameTypeOperands]> :
    Crab_Op<mnemonic, traits>,
    Arguments<(ins CrabPredicateAttr:$predicate, 
            CrabInteger:$lhs, 
            CrabInteger:$rhs)> {
    
    let description = [{
        The first argument is an attribute that defines which type of comparison is
        performed. The following comparisons are supported:

        -   equal (mnemonic: `"eq"`; integer value: `0`)
        -   not equal (mnemonic: `"ne"`; integer value: `1`)
        -   signed less than (mnemonic: `"slt"`; integer value: `2`)
        -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
        -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
        -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
        -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
        -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
        -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
        -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    }];

    let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";

    let parser = [{
        return impl::parseOneResultSameOperandTypeOp(parser, result);
    }];

    let extraClassDeclaration = [{
        static StringRef getPredicateAttrName() { return "predicate"; }
        static CrabPredicate getPredicateByName(StringRef name);

        CrabPredicate getPredicate() {
        return (CrabPredicate)(*this)->getAttrOfType<IntegerAttr>(
            getPredicateAttrName()).getInt();
        }
    }];

    let verifier = [{ return success(); }];

    let assemblyFormat = "$predicate `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

def AssertOp : CrabConstraintOp<"assert"> {
  let summary = "crab assertion";
  let description = [{
    The `assert` operation is an assertion tied to a  generic comparison
    for two arguments that need to have their types matching.

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = crab.assert ne(%lhs, %rhs) : i32
    ```
  }];
} 

def AssumeOp : CrabConstraintOp<"assume"> {
  let summary = "crab assume";
  let description = [{
    The `assume` operation is a constraint tied to a  generic comparison
    for two arguments that need to have their types matching.

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = crab.assume sle(%lhs, %rhs) : i32
    ```
  }];
} 

//===----------------------------------------------------------------------===//
// crab constant ops 
//===----------------------------------------------------------------------===//

def ConstantOp : Crab_Op<"const", [ConstantLike, NoSideEffect]> {
  let summary = "integer constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute. 

    Example:

    ```mlir
    // Integer constant
    %1 = constant 42 : i32
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "Attribute":$value, "Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let parser = [{
    return parseConstantOp(parser, result);
  }];

  let printer = [{
    return printConstantOp(p, *this);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// crab havoc ops 
//===----------------------------------------------------------------------===//

def HavocOp : Crab_Op<"havoc"> {
    let summary = "crab havoc";
    let description = [{
        This operation takes no input and returns a CrabInteger

        Example:

        ```mlir
        // invoke the havoc operation
        %0 = crab.havoc() : i32
        ```
    }];

    let results = (outs CrabInteger:$result);

    let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// crab br ops 
//===----------------------------------------------------------------------===//

def UnaryBranchOp : Crab_Op<"br",
    [//JN: cannot compile this line
     //DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "unary branch operation";
  let description = [{
    The `br` operation represents a branch operation in a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types for each successor must match the arguments of
    the block successor.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      br ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "Block *":$dest,
                  CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  // BranchOp is fully verified by traits.
  let verifier = ?;

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  //let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}
  
def BinaryBranchOp : Crab_Op<"nd_br",
  [AttrSizedOperandSegments, NoSideEffect, Terminator]> {
  let summary = "binary branch operation";
  let description = [{
    The `nd_br` terminator operation represents a nondeterministic branch. We 
    nondeterministically select one of the possible targets.

    This branch operation is not allowed to target the entry block for a region. 
    The two destinations of the conditional branch operation are allowed to be the same.

    Example:

    ```mlir
    func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      crab.br ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$lhsDestOperands,
                    Variadic<AnyType>:$rhsDestOperands);
  let successors = (successor AnySuccessor:$lhsDest, AnySuccessor:$rhsDest);

  let builders = [
    OpBuilder<(ins "Block *":$lhsDest, "ValueRange":$lhsOperands, 
        "Block *":$rhsDest, "ValueRange":$rhsOperands), 
    [{
        build($_builder, $_state, lhsOperands, rhsOperands, lhsDest, rhsDest);
    }]>];

  let assemblyFormat = [{
    $lhsDest (`(` $lhsDestOperands^ `:` type($lhsDestOperands) `)`)? `,`
    $rhsDest (`(` $rhsDestOperands^ `:` type($rhsDestOperands) `)`)?
    attr-dict
  }];
}


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Crab_Op<"call",
    [CallOpInterface , //MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call. The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

let builders = [
    // JN: cannot compile due to unknown symbol FuncOp
    // OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
    //   $_state.addOperands(operands);
    //   $_state.addAttribute("callee", SymbolRefAttr::get(callee));
    //   $_state.addTypes(callee.getType().getResults());
    // }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Crab_Op<"return",
[NoSideEffect,
 // JN: cannot compile due to unknown symbol FuncOp
 // HasParent<"FuncOp">,
 //MemRefsNormalizable,
 ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{
    The `return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    func @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}


#endif // CRAB_OPS
