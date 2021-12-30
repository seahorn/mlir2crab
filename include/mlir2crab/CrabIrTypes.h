#pragma once

/* This file defines all CrabIR types */

#include <crab/cfg/basic_block_traits.hpp>
#include <crab/cfg/cfg.hpp>
#include <crab/cg/cg.hpp>
#include <crab/config.h>
#include <crab/support/debug.hpp>
#include <crab/types/varname_factory.hpp>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Block.h>

#include <functional> // for hash
#include <string>

namespace std {
template <> struct hash<mlir::Value> {
  size_t operator()(mlir::Value v) const {
    return mlir::hash_value(v);
  }
};
template <> struct hash<const mlir::Block*> {
  size_t operator()(const mlir::Block *b) const {
    return std::hash<void*>{}((void*)b);
  }
};  
} // end namespace std


namespace mlir2crab {
// We can use "const mlir::Block*" as Crab basic block label.
// However, Crab CFG does not allow to attach statements to CFG edges
// but we need to do that in order to model the semantics of MLIR
// block parameters.  The idea is that given a branch instruction to
// bb(%x, %y) at block A and being bb defined as bb(%1,%2) in block B
// then we create a new Crab block between A and B where we add the
// assignments "%1 := %x" and "%2 := %y".
class block_label {
  const mlir::Block* m_bb;
  std::pair<const mlir::Block*, const mlir::Block*> m_edge;  
public:
  block_label(): m_bb(nullptr), m_edge({nullptr, nullptr}) {}
  // block label is a mlir::Block*  
  block_label(const mlir::Block* bb)
    : m_bb(bb), m_edge(nullptr, nullptr) {}
  // block label is an edge between src and dst
  block_label(const mlir::Block* src, const mlir::Block* dst)
    : m_bb(nullptr), m_edge({src,dst}) {}
  
  bool operator==(const block_label &o) const {    
    if (m_bb && o.m_bb) {
      return m_bb == o.m_bb;
    } else if (!m_bb && !o.m_bb) {
      return (m_edge.first == o.m_edge.first &&
	      m_edge.second == o.m_edge.second);
    } else {
      return false;
    }
  }
  bool operator!=(const block_label &o) const {
    return !(*this == o);
  }
  bool operator<(const block_label &o) const {
    if (m_bb && o.m_bb) {
      return m_bb < o.m_bb;
    } else if (!m_bb && !o.m_bb) {
      if (m_edge.first < o.m_edge.first) {
	return true;
      } else if (m_edge.first > o.m_edge.first) {
	return false;
      } else {
	return m_edge.second < o.m_edge.second;
      }
    } else {
      return false;
    }    
  }

  bool is_edge() const {
    return !m_bb;
  }

  // it returns null if is_edge() returns true
  const mlir::Block* get_block() const {
    return m_bb;
  }

  // it returns a pair of null pointers if is_edge() returns false
  std::pair<const mlir::Block*, const mlir::Block*> get_edge() const {
    return m_edge;
  }
  
  std::size_t hash() const {
    if (m_bb) {
      return std::hash<const mlir::Block*>{}(m_bb);
    } else {
      return boost::hash_value(m_edge);
    }
  }
  std::string to_string() const {
    auto block_toString = [](const mlir::Block* b) -> std::string {
      std::string str;
      llvm::raw_string_ostream os(str);
      // UNSAFE but printAsOperand is not marked as const
      (const_cast<mlir::Block*>(b))->printAsOperand(os);
      return os.str();        
    };
    
    if (m_bb) {
      return block_toString(m_bb);      
    } else {
      return (std::string("edge_") + block_toString(m_edge.first) +
	      std::string("_") + block_toString(m_edge.second));	
    }
  }

  friend ::crab::crab_os &operator<<(::crab::crab_os &o, const block_label &bl) {
    o << bl.to_string();
    return o;
  }
  
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const block_label &bl) {
    o << bl.to_string();
    return o;
  }				   
};

// Wrapper to mlir::Value to assign names.
// We could use mlir::Value directly as parameter to variable_factory
// but I don't know how to extract string names from mlir::Value's.  
class named_value {
  mlir::Value m_v;  
  unsigned m_id;
public:
  named_value(mlir::Value v, unsigned id)
    : m_v(v), m_id(id) {}

  std::string to_string() const {
    return std::string("%") + std::to_string(m_id);
  }
  
  // Needed by variable_factory
  bool operator==(const named_value&other) const {
    return m_v == other.m_v;
  }
  size_t hash() const {
    return mlir::hash_value(m_v);
  }
};  
} //end namespace mlir2crab


namespace std {
template <> struct hash<mlir2crab::block_label> {
  size_t operator()(const mlir2crab::block_label &b) const {
    return b.hash();
  }
};
template <> struct hash<mlir2crab::named_value> {
  size_t operator()(const mlir2crab::named_value &v) const {
    return v.hash();
  }
};  
} // namespace std

namespace mlir2crab {
// Crab variable factory
using variable_factory_t = crab::var_factory_impl::variable_factory<named_value>;
// Crab variable names
using varname_t = typename variable_factory_t::varname_t; 
// Crab basic block labels
using block_label_t = block_label;
// Crab numbers
using number_t = ikos::z_number;
// Crab CFG types
using cfg_t = crab::cfg::cfg<block_label_t, varname_t, number_t>;
using cfg_ref_t = crab::cfg::cfg_ref<cfg_t>;
using block_t = cfg_t::basic_block_t;
// Crab variable and expressions
using variable_t = crab::variable<number_t, varname_t>;
using variable_or_constant_t = crab::variable_or_constant<number_t, varname_t>;
using linear_expression_t = ikos::linear_expression<number_t, varname_t>;
using linear_constraint_t = ikos::linear_constraint<number_t, varname_t>;
using reference_constraint_t = crab::reference_constraint<number_t, varname_t>;
// Crab callgraph types
using callgraph_t = crab::cg::call_graph<cfg_ref_t>;
using callgraph_ref_t = crab::cg::call_graph_ref<callgraph_t>;
} // end namespace mlir2crab

namespace crab {
template <> class variable_name_traits<mlir2crab::named_value> {
public: 
static std::string to_string(mlir2crab::named_value v) {
  return v.to_string();  
}
};
template <> class basic_block_traits<mlir2crab::block_t> {
public:
static std::string to_string(const mlir2crab::block_label_t &b) {
  return b.to_string();
}
};
} // end namespace crab
