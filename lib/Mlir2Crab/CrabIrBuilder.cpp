#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir2crab/Dialect/Crab/IR/CrabDialect.h"
#include "mlir2crab/mlir2crab.h"
#include "mlir2crab/CrabIrBuilderOpts.h"
#include "mlir2crab/Support/Log.h"
#include "mlir2crab/Support/Error.h"
#include "mlir2crab/CrabIrTypes.h"

#include <functional>
#include <memory>

using namespace mlir;
using namespace llvm;
  
namespace mlir2crab {
class CrabIrAnalyzerImpl;

class CrabIrBuilderImpl {
  OwningModuleRef m_module;  
  const CrabIrBuilderOpts &m_opts;
  llvm::DenseMap<mlir::Value, variable_or_constant_t> m_cache;  
  /** Begin crab **/
  variable_factory_t m_vfac;
  unsigned m_varname_id;
  llvm::DenseMap<const mlir::Operation*, ::std::unique_ptr<cfg_t>> m_cfgs;  
  block_t *m_current_block;
  /** End crab **/
  // for debug messages
  mlir::Operation *m_current_operation;
  std::unique_ptr<callgraph_t> m_cg;
  
  /** begin helpers **/
  static StringRef getOperationName(Operation &op) {
    return op.getName().getIdentifier();
  }

  static bool isBinaryOp(mlir::Operation &stmt) {
    if (stmt.getNumOperands() == 2 && stmt.getNumResults() == 1) {      
      StringRef opName = getOperationName(stmt);
      return 
      (opName == "crab.add"  || opName == "crab.sub"  || opName == "crab.mul" || 
       opName == "crab.sdiv" || opName == "crab.udiv" ||
       opName == "crab.srem" || opName == "crab.urem" ||
       opName == "crab.and"  || opName == "crab.or"   || opName == "crab.xor"); 
    }
    return false;
  }

  static bool isCstOp(mlir::Operation &stmt) {
    return ((getOperationName(stmt) == "crab.const") &&
	    (stmt.getNumResults() == 1) && 
	    (stmt.getNumOperands() == 0));
  }

  static number_t toCrabNumber(const mlir::APInt &v) {  
    // Based on:
    // https://llvm.org/svn/llvm-project/polly/trunk/lib/Support/GICHelper.cpp
    mlir::APInt abs;
    abs = v.isNegative() ? v.abs() : v;
    const uint64_t *rawdata = abs.getRawData();
    unsigned numWords = abs.getNumWords();    
    number_t res;
    // FIXME: assume number_t has get_mpz_t() method
    mpz_import(res.get_mpz_t(), numWords, -1, sizeof(uint64_t), 0, 0, rawdata);
    return v.isNegative() ? -res : res;
  }
  /** end helpers **/
      
  cfg_t& getCFG(const mlir::Block& block);
  block_t& lookup(const mlir::Block& block);
  const variable_or_constant_t& lookup(const mlir::Value &v);
  
  void translateFunction(mlir::Operation &function);
  void translateBlock(mlir::Block &block);
  void translateStatement(mlir::Operation &statement);

public:
  CrabIrBuilderImpl(OwningModuleRef &&module, const CrabIrBuilderOpts &opts);
  void generate();
  const CrabIrBuilderOpts& getOpts() const {
    return m_opts;
  }  
  const callgraph_t &getCallGraph() const;
  callgraph_t &getCallGraph();  
};

CrabIrBuilderImpl::CrabIrBuilderImpl(OwningModuleRef &&module,
                                     const CrabIrBuilderOpts &opts)
  : m_module(::std::move(module)), m_opts(opts),
    m_varname_id(0),
    m_current_block(nullptr), m_current_operation(nullptr), m_cg(nullptr) {
#if 0 
  m_module->print(MSG);
#endif  
}

void CrabIrBuilderImpl::generate() {
  // Translate module
  mlir::Block *body = m_module->getBody();        
  for (mlir::Operation &op: body->getOperations()) {
    translateFunction(op);
  }    
#if 1
  for (auto &kv: m_cfgs) {
    ::crab::outs() << *(kv.second) << "\n";
  }
#endif

  // Generate callgraph
  std::vector<cfg_ref_t> cfgs;
  for (auto &kv: m_cfgs) {
    cfgs.push_back(*(kv.second));
  }
  m_cg = std::make_unique<callgraph_t>(cfgs);

  INFO << "finished translation to CrabIR.\n";   
}

const callgraph_t& CrabIrBuilderImpl::getCallGraph() const {
  if (!m_cg) {
    ERROR_AND_ABORT("call generate() before getCallGraph()");
  }
  return *m_cg;
}

callgraph_t& CrabIrBuilderImpl::getCallGraph() {
  if (!m_cg) {
    ERROR_AND_ABORT("call generate() before getCallGraph()");
  }
  return *m_cg;
}  
   
cfg_t& CrabIrBuilderImpl::getCFG(const mlir::Block &block) {
  const mlir::Operation *func = (const_cast<mlir::Block*>(&block))->getParentOp();
  auto it = m_cfgs.find(func);
  if (it != m_cfgs.end()) {
    return *(it->second);
  }  
  ERROR_AND_ABORT("Crab cfg not found");
}

// Map a mlir block to a Crab block  
block_t& CrabIrBuilderImpl::lookup(const mlir::Block &block) {
  cfg_t &cfg = getCFG(block);
  block_label bl(&block);
  return cfg.get_node(bl);
}

// Map a mlir value to a Crab variable or constant
const variable_or_constant_t& CrabIrBuilderImpl::lookup(const mlir::Value &v) {    
  auto it = m_cache.find(v);
  if (it != m_cache.end()) {
    return it->second;
  }

  // Constant
  Operation *DefOp = v.getDefiningOp();
  if (DefOp && getOperationName(*DefOp) == "crab.const") {
    IntegerAttr attr = DefOp->getAttr("value").cast<IntegerAttr>();
    mlir::Type type = attr.getType();
    if (IntegerType itype = type.dyn_cast<IntegerType>()) {      
      unsigned bitwidth = itype.getWidth();
      mlir::APInt val = attr.getValue();
      variable_or_constant_t res(toCrabNumber(val),
				 ::crab::variable_type(::crab::INT_TYPE, bitwidth));
      auto it = m_cache.insert(std::make_pair(v, res)).first;
      return it->second;
    }
  }

  // Variable
  if (IntegerType ity = v.getType().dyn_cast<IntegerType>()) {      
    unsigned bitwidth = ity.getWidth();
    named_value nv(v, m_varname_id++);
    variable_t res(m_vfac[nv], ::crab::INT_TYPE, bitwidth);
    it = m_cache.insert(std::make_pair(v, res)).first;
    return it->second;
  }
  
  ERROR_AND_ABORT("lookup with unexpected " << v);
}

  
void CrabIrBuilderImpl::translateFunction(mlir::Operation &function) { 
  mlir::Region &body = function.getRegions()[0];
  mlir::Block *entryBlock = nullptr;
  
  // JN: I don't know how to get the entry block without iterating
  for (mlir::Block &block: body.getBlocks()) {
    if (block.isEntryBlock()) {
      entryBlock = &block;
      break;
    }
  }
  if (!entryBlock) {
    ERROR_AND_ABORT("No entry block found");
  }  
  // Create crab CFG
  std::unique_ptr<cfg_t> cfg = std::make_unique<cfg_t>(entryBlock);
  using func_decl_t = typename cfg_t::fdecl_t;
  // TODO: add function parameters
  StringRef func_name = function.getAttrOfType<mlir::StringAttr>("sym_name").getValue();
  func_decl_t func_decl(func_name.str(), {},{});
  cfg->set_func_decl(func_decl);

  // Create first all Crab basic blocks
  for (mlir::Block &block: body.getBlocks()) {
    block_label bl(&block);
    cfg->insert(bl);
  }
  m_cfgs.insert(std::make_pair(&function, ::std::move(cfg)));

  // Translate the content of each block
  for (mlir::Block &block: body.getBlocks()) {
    translateBlock(block);    
  }  
}

void CrabIrBuilderImpl::translateBlock(mlir::Block &block) { 
#if 0  
  block.print(MSG);
#endif   
  m_current_block = &(lookup(block));
  for (mlir::Operation &op: block.getOperations()) {
    translateStatement(op);
  }
  m_current_block = nullptr;
}


#define CONMMUTATIVE_OP(OP,RES,OP1,OP2)					\
  if (OP1.is_variable()) {						\
    if (OP2.is_variable()) {						\
      m_current_block->OP(RES.get_variable(), OP1.get_variable(), OP2.get_variable()); \
    } else {								\
      m_current_block->OP(RES.get_variable(), OP1.get_variable(), OP2.get_constant()); \
    }									\
  } else {								\
    if (OP2.is_variable()) {						\
      m_current_block->OP(RES.get_variable(), OP2.get_variable(), OP1.get_constant()); \
    } else {								\
      ERROR_AND_ABORT("Crab does not support arithmetic operation with both constant operands"); \
    }									\
  }

#define NON_CONMMUTATIVE_OP(OP,RES,OP1,OP2)				\
  if (OP1.is_variable()) {						\
    if (OP2.is_variable()) {						\
      m_current_block->OP(RES.get_variable(), OP1.get_variable(), OP2.get_variable()); \
    } else {								\
      m_current_block->OP(RES.get_variable(), OP1.get_variable(), OP2.get_constant()); \
    }									\
  } else {								\
    if (OP2.is_variable()) {						\
      ERROR_AND_ABORT("Crab does not support a constant 1st operand");	\
    } else {								\
      ERROR_AND_ABORT("Crab does not support arithmetic operation with both constant operands"); \
    }									\
  }   


void CrabIrBuilderImpl::translateStatement(mlir::Operation &statement) {
  auto to_lin_exp = [](const variable_or_constant_t &x) -> linear_expression_t {
    if (x.is_variable()) {
      return linear_expression_t(x.get_variable());
    } else {
      return linear_expression_t(x.get_constant());
    }
  };
  
  m_current_operation = &statement;
  StringRef opName(getOperationName(statement));
    
  if (opName == "crab.havoc") {
    variable_or_constant_t res = lookup(statement.getResults()[0]);
    m_current_block->havoc(res.get_variable());
  } else if (isCstOp(statement)) {
    // do nothing: translated elsewhere
  } else if (opName == "crab.call") {
    // TODO: translation of calls
    variable_or_constant_t res = lookup(statement.getResults()[0]);
    m_current_block->havoc(res.get_variable());
  } else if (isBinaryOp(statement)) {                    
    variable_or_constant_t op1 = lookup(statement.getOperands()[0]);
    variable_or_constant_t op2 = lookup(statement.getOperands()[1]);
    variable_or_constant_t res = lookup(statement.getResults()[0]);    
    if (opName == "crab.add") {
      CONMMUTATIVE_OP(add, res, op1, op2);
    } else if (opName == "crab.sub") {
      NON_CONMMUTATIVE_OP(sub, res, op1, op2);
    } else if (opName == "crab.mul") {
      NON_CONMMUTATIVE_OP(mul, res, op1, op2);
    } else if (opName == "crab.sdiv") {
      NON_CONMMUTATIVE_OP(div, res, op1, op2);
    } else if (opName == "crab.udiv") {
      NON_CONMMUTATIVE_OP(udiv, res, op1, op2);
    } else if (opName == "crab.srem") {
      NON_CONMMUTATIVE_OP(rem, res, op1, op2);
    } else if (opName == "crab.urem") {
      NON_CONMMUTATIVE_OP(urem, res, op1, op2);
    } else if (opName == "crab.and") {
      CONMMUTATIVE_OP(bitwise_and, res, op1, op2);      
    } else if (opName == "crab.or") {
      CONMMUTATIVE_OP(bitwise_or, res, op1, op2);
    } else if (opName == "crab.xor") {
      CONMMUTATIVE_OP(bitwise_xor, res, op1, op2);     
    } 
  } else if (opName == "crab.br") {
    mlir::Block *currB = statement.getBlock();
    assert(currB->getNumSuccessors() == 1);   
    mlir::Block *succB = currB->getSuccessors()[0];    
    
    // Add assignments for modeling semantics of block arguments
    for (unsigned i=0, sz=statement.getNumOperands(); i<sz;i++) {      
      variable_or_constant_t lhs = lookup(succB->getArguments()[i]);
      variable_or_constant_t rhs = lookup(statement.getOperands()[i]);
      assert(lhs.is_variable());
      // integer assignment
      m_current_block->assign(lhs.get_variable(), to_lin_exp(rhs));
    }

    // Add CFG edge between current and successor
    m_current_block->add_succ(lookup(*succB));        
  } else if (opName == "crab.nd_br") {
    mlir::Block *currB = statement.getBlock();
    assert(currB->getNumSuccessors() == 2);   
    mlir::Block *succLeftB = currB->getSuccessors()[0];
    mlir::Block *succRightB = currB->getSuccessors()[1];

    cfg_t &cfg = getCFG(*currB);
    block_label edgeLeft(currB, succLeftB);
    block_label edgeRight(currB, succRightB);    
    block_t &leftB = cfg.insert(edgeLeft);
    block_t &rightB = cfg.insert(edgeRight);

    // Add assignments for modeling semantics of block arguments    
    unsigned numLeftArgs = succLeftB->getArguments().size(); 
    for (unsigned i=0, sz=numLeftArgs; i<sz;i++) {
      variable_or_constant_t lhs = lookup(succLeftB->getArguments()[i]);
      variable_or_constant_t rhs = lookup(statement.getOperands()[i]);
      assert(lhs.is_variable());
      // integer assignment
      leftB.assign(lhs.get_variable(), to_lin_exp(rhs));      
    }

    for (unsigned i=0, sz=succRightB->getArguments().size(); i<sz;i++) {
      variable_or_constant_t lhs = lookup(succRightB->getArguments()[i]);
      variable_or_constant_t rhs = lookup(statement.getOperands()[numLeftArgs+i]);
      assert(lhs.is_variable());
      // integer assignment
      rightB.assign(lhs.get_variable(), to_lin_exp(rhs));      
    }               
    
    // Add CFG edges between current and successors
    m_current_block->add_succ(leftB);
    m_current_block->add_succ(rightB);
    leftB.add_succ(lookup(*succLeftB));
    rightB.add_succ(lookup(*succRightB));     
  } else if (opName == "crab.assume" || opName == "crab.assert") {
    variable_or_constant_t op1 = lookup(statement.getOperands()[0]);
    variable_or_constant_t op2 = lookup(statement.getOperands()[1]);
    linear_constraint_t cst;
    auto predicateCode = statement.getAttr("predicate").cast<IntegerAttr>();
    switch(predicateCode.getValue().getZExtValue()) {
    case 0: /*  eq */
      cst = linear_constraint_t(to_lin_exp(op1) == to_lin_exp(op2));
      break;
    case 1: /*  ne */
      cst = linear_constraint_t(to_lin_exp(op1) != to_lin_exp(op2));
      break;      
    case 2: /*  slt */
      cst = linear_constraint_t(to_lin_exp(op1) < to_lin_exp(op2));
      cst.set_signed();
      break;      
    case 3: /*  sle */
      cst = linear_constraint_t(to_lin_exp(op1) <= to_lin_exp(op2));
      cst.set_signed();      
      break;      
    case 4: /*  sgt */
      cst = linear_constraint_t(to_lin_exp(op1) > to_lin_exp(op2));
      cst.set_signed();      
      break;      
    case 5: /*  sge */
      cst = linear_constraint_t(to_lin_exp(op1) >= to_lin_exp(op2));
      cst.set_signed();      
      break;      
    case 6: /*  ult */
      cst = linear_constraint_t(to_lin_exp(op1) < to_lin_exp(op2));
      cst.set_unsigned();
      break;      
    case 7: /*  ule */
      cst = linear_constraint_t(to_lin_exp(op1) <= to_lin_exp(op2));
      cst.set_unsigned();      
      break;      
    case 8: /*  ugt */
      cst = linear_constraint_t(to_lin_exp(op1) > to_lin_exp(op2));
      cst.set_unsigned();      
      break;      
    case 9: /*  uge */
      cst = linear_constraint_t(to_lin_exp(op1) >= to_lin_exp(op2));
      cst.set_unsigned();      
      break;      
    }
    if (opName == "crab.assume") {
      m_current_block->assume(cst);
    } else {
      // TODO: add crab debug info 
      m_current_block->assertion(cst);
    }
  } else {
    WARN << "TODO translation of " << statement << "\n";
  }         
  m_current_operation = nullptr;
}

CrabIrBuilder::CrabIrBuilder(OwningModuleRef &&module,
                             const CrabIrBuilderOpts &opts)
  : m_impl(new CrabIrBuilderImpl(::std::move(module), opts)) {}

CrabIrBuilder::~CrabIrBuilder() {
}

const CrabIrBuilderOpts& CrabIrBuilder::getOpts() const {
  return m_impl->getOpts();
}  
void CrabIrBuilder::generate() {
  m_impl->generate();
}

const callgraph_t& CrabIrBuilder::getCallGraph() const {
  return m_impl->getCallGraph();
}

callgraph_t& CrabIrBuilder::getCallGraph() {
  return m_impl->getCallGraph();
}
   
} // end namespace mlir2crab
