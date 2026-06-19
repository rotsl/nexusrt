// =============================================================================
// NexusRT — src/pipeline/postprocess.cpp
// =============================================================================
#include "pipeline/postprocess.hpp"
#include <sstream>

namespace nexusrt { namespace pipeline {
Status PostprocessStage::run(const std::vector<int32_t>& tokens,
                             std::string& out_text, bool copy_to_host) {
    (void)ctx_;  // detokenization kernel would run here
    std::ostringstream os;
    for (auto t : tokens) {
        if (t >= 32 && t < 127) os << static_cast<char>(t);
        else os << ' ';
    }
    out_text = os.str();
    (void)copy_to_host;  // GPU-side text always copied to host here
    return Status::Ok;
}
}} // namespace
