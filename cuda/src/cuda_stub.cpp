#include "evaluation.h"
#include <stdexcept>
namespace qbcs {
std::unique_ptr<BatchEvaluator> make_cuda_evaluator(const Encoding&, const EvaluationOptions&) {
    throw std::runtime_error("This executable was built without CUDA. Rebuild with -DQBCS_ENABLE_CUDA=ON, or explicitly use --backend cpu / --backend emulated.");
}
} // namespace qbcs
