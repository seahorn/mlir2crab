#include "mlir2crab/Support/Log.h"

using namespace llvm;
bool mlir2crab::WarnEnable = true;

namespace mlir2crab {
void SetWarn(bool v) { WarnEnable = v; }

warn_ostream::warn_ostream(raw_ostream &OS, const std::string &prefix,
                           const std::string &suffix)
  : raw_svector_ostream(m_buffer), m_os(OS), m_prefix(prefix),
    m_suffix(suffix) {}
  
warn_ostream::~warn_ostream() {
  if (!WarnEnable) {
    return;
  }
  m_os << m_prefix;
  m_os << str();
#if !defined(NDEBUG)
  if (!m_suffix.empty())
    m_os << " (" << m_suffix << ")";
#endif
  m_os << "\n";
  resetColor();
}
} // end namespace mlir2crab
