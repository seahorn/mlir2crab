#pragma once

/* This file defines all Crab domain types */
#include <mlir2crab/CrabIrTypes.h>

#include <crab/config.h>
#include <crab/domains/generic_abstract_domain.hpp>
#include <crab/domains/intervals.hpp>
#include <crab/domains/graphs/graph_config.hpp>
#include <crab/domains/split_dbm.hpp>
#include <crab/domains/split_oct.hpp>

namespace mlir2crab {
// A wrapper for an arbitrary abstract domain, cheap to copy
using crab_abstract_domain = ::crab::domains::abstract_domain_ref<variable_t>;

using interval_domain_t = ikos::interval_domain<number_t, varname_t>;

/// To choose DBM parameters
struct BigNumDBMParams {
  /* This version uses mathematical integers so no overflow */  
  using Wt = ikos::z_number;
  using graph_t = ::crab::SparseWtGraph<Wt>;
};
struct SafeFastDBMParams {
  /* This version checks for overflow and raise error if detected*/  
  using Wt = ::crab::safe_i64;
  using graph_t = ::crab::AdaptGraph<Wt>;
};
struct FastDBMParams {
  /* This version does not check for overflow */
  using Wt = int64_t;
  using graph_t = ::crab::AdaptGraph<Wt>;
};

using zones_domain_t = ::crab::domains::split_dbm_domain<number_t, varname_t, SafeFastDBMParams>;
using oct_domain_t = ::crab::domains::split_oct_domain<number_t, varname_t, SafeFastDBMParams>;  
} // end namespace mlir2crab
