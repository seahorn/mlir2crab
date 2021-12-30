#include "mlir2crab/mlir2crab.h"
#include "mlir2crab/CrabIrTypes.h"
#include "mlir2crab/CrabDomainTypes.h"
#include "mlir2crab/Support/Error.h"
#include "mlir2crab/Support/Log.h"

#include "crab/analysis/inter/inter_params.hpp"
#include "crab/analysis/inter/top_down_inter_analyzer.hpp"
#include "crab/checkers/base_property.hpp"
#include "crab/cg/cg.hpp"
#include "crab/cg/cg_bgl.hpp"
#include "crab/support/stats.hpp"
#include "crab/support/os.hpp"

#include "llvm/Support/raw_ostream.h"

#include <map>

namespace mlir2crab {

class DomainRegistry {
  using domainKey = Domain;
public:
  using FactoryMap = std::map<domainKey, crab_abstract_domain>;

  template <typename AbsDom> static bool add(domainKey dom_ty) {
    auto &map = getFactoryMap();
    auto dom = DomainRegistry::makeTopDomain<AbsDom>();
    bool res = map.insert({dom_ty, dom}).second;
    crab::CrabStats::reset();
    return res;
  }

  static bool count(domainKey dom_ty) {
    auto &map = getFactoryMap();
    return map.find(dom_ty) != map.end();
  }

  static crab_abstract_domain at(domainKey dom_ty) {
    auto &map = getFactoryMap();
    return map.at(dom_ty);
  }

private:
  static FactoryMap &getFactoryMap() {
    static FactoryMap map;
    return map;
  }

  template <typename AbsDom> static crab_abstract_domain makeTopDomain() {
    AbsDom dom_val;
    crab_abstract_domain res(std::move(dom_val));
    return res;
  }
}; // end namespace DomainRegistry

#define REGISTER_DOMAIN(domain_enum_val, domain_decl)                          \
  bool domain_decl##_entry = DomainRegistry::add<domain_decl>(domain_enum_val);  
  
class CrabIrAnalyzerImpl {
 using inter_params_t =
    ::crab::analyzer::inter_analyzer_parameters<callgraph_t>;
  using inter_analyzer_t =
    ::crab::analyzer::top_down_inter_analyzer<callgraph_t, crab_abstract_domain>;
  using checks_db_t = ::crab::checker::checks_db;
    
  CrabIrBuilder &m_crabIR;
  const CrabIrAnalyzerOpts &m_opts;
  std::unique_ptr<inter_analyzer_t> m_crabAnalyzer;
  checks_db_t m_checks;
  
public:
  CrabIrAnalyzerImpl(CrabIrBuilder &crabIR, const CrabIrAnalyzerOpts &opts)
    : m_crabIR(crabIR), m_opts(opts), m_crabAnalyzer(nullptr) {}

  const CrabIrAnalyzerOpts& getOpts() const {
    return m_opts;
  }  
  void analyze();
  void write(llvm::raw_ostream&os) const;
}; //end namespace CrabIrAnalyzerImpl


CrabIrAnalyzer::CrabIrAnalyzer(CrabIrBuilder &crabIR, const CrabIrAnalyzerOpts &opts)
  : m_impl(new CrabIrAnalyzerImpl(crabIR, opts)) {
}

CrabIrAnalyzer::~CrabIrAnalyzer() {}

const CrabIrAnalyzerOpts& CrabIrAnalyzer::getOpts() const {
  return m_impl->getOpts();
}
  
void CrabIrAnalyzer::analyze() {
  m_impl->analyze();
}

void CrabIrAnalyzer::write(llvm::raw_ostream &os) const {
  m_impl->write(os);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream &os, const CrabIrAnalyzer &analyzer) {
  analyzer.write(os);
  return os;
}
  
void CrabIrAnalyzerImpl::analyze() {  
  if (DomainRegistry::count(m_opts.domain)) {
    inter_params_t params;
    params.run_checker = m_opts.run_checker;    
    crab_abstract_domain init = DomainRegistry::at(m_opts.domain);
    m_crabAnalyzer = std::make_unique<inter_analyzer_t>(m_crabIR.getCallGraph(), init, params);
    m_crabAnalyzer->run(init);
    if (m_opts.run_checker) {
      m_checks += m_crabAnalyzer->get_all_checks();
    }    
  } else {
    ERROR_AND_ABORT("Abstract domain not recognized");
  }  
}

void CrabIrAnalyzerImpl::write(llvm::raw_ostream &os) const {
  if (!m_crabAnalyzer) {
    WARN << "Call analyze() before write invariants";
  } else {
    m_crabIR.getOpts().write(os);
    os << "\n";
    getOpts().write(os);
    os << "\n";
    os << "=== Verification results === \n";
    ::crab::crab_string_os crab_os;
    m_checks.write(crab_os);
    os << crab_os.str() << "\n";
    os << "=== Invariants === \n";
    os << "TODO\n";
  }
}

  
REGISTER_DOMAIN(Domain::Intervals, interval_domain_t)
REGISTER_DOMAIN(Domain::Zones, zones_domain_t)
REGISTER_DOMAIN(Domain::Octagons, oct_domain_t)

} // end namespace mlir2crab
