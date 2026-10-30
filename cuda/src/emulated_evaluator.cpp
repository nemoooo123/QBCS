// CPU execution of the actual integer primitives used by the CUDA thread kernel.
// This is a verification backend, not a GPU emulator or a performance prediction.
#include "evaluation.h"
#include "batch_layout.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qbcs {
namespace {
class EmulatedEvaluator final : public BatchEvaluator {
public:
    EmulatedEvaluator(const Encoding& e, const EvaluationOptions& o) : layout_(e) {
        if (o.batch_capacity < 1 || o.memory_mb > std::numeric_limits<size_t>::max() / (1024 * 1024))
            throw std::invalid_argument("invalid emulated evaluator capacity/memory budget");
        capacity_ = o.batch_capacity;
        if (o.memory_mb) {
            const size_t budget = o.memory_mb * size_t(1024 * 1024);
            if (budget <= layout_.metadata_bytes()) throw std::runtime_error("emulated memory budget is too small");
            capacity_ = int(std::min<size_t>(capacity_, (budget - layout_.metadata_bytes()) / layout_.bytes_per_sample()));
        }
        if (capacity_ < 1) throw std::runtime_error("memory budget cannot hold one sample");
        input_.resize(std::max<size_t>(1, layout_.input_bytes * capacity_));
        schedule_.resize(size_t(layout_.active_steps) * capacity_);
        offsets_.resize(schedule_.size());
        counts_.resize(capacity_);
        edges_.resize(size_t(layout_.max_gates) * capacity_);
        previous0_.resize(edges_.size()); previous1_.resize(edges_.size());
        heads_.resize(layout_.vertices * capacity_);
    }
    int capacity() const override { return capacity_; }
    std::string description() const override {
        std::ostringstream s;
        s << "emulated integer core (CPU only), capacity=" << capacity_
          << ", scratch bytes/sample=" << layout_.bytes_per_sample();
        return s.str();
    }
    void evaluate(const std::vector<const Sample*>& samples, std::vector<int>& fitness) override {
        const size_t batch = samples.size();
        if (batch > size_t(capacity_)) throw std::runtime_error("batch exceeds emulator capacity");
        fitness.resize(batch);
        if (!batch) return;
        if (!layout_.active_steps) {
            std::fill(fitness.begin(), fitness.end(), 0);
            ++batches_; samples_ += batch;
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        layout_.pack(samples, input_.data(), schedule_.data(), offsets_.data(), counts_.data());
        std::fill_n(heads_.data(), layout_.vertices * batch, 0u);
        for (int k = 0; k < layout_.active_steps; ++k)
            for (size_t i = 0; i < batch; ++i) {
                const size_t slot = size_t(k) * batch + i;
                const uint8_t* row = input_.data() + i * layout_.input_bytes;
                gpu_core::decode_step(layout_.steps[schedule_[slot]], row, row + layout_.encoding.l3_size,
                    edges_.data() + size_t(offsets_[slot]) * batch + i, batch);
            }
        for (size_t i = 0; i < batch; ++i)
            fitness[i] = gpu_core::reduce(edges_.data() + i, counts_[i], batch,
                heads_.data() + i, previous0_.data() + i, previous1_.data() + i);
        seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++batches_; samples_ += batch;
    }
    std::string statistics() const override {
        std::ostringstream s;
        s << "emulated batches=" << batches_ << ", samples=" << samples_ << ", service compute seconds=" << seconds_;
        return s.str();
    }
private:
    BatchLayout layout_;
    int capacity_ = 1;
    std::vector<uint8_t> input_;
    std::vector<int> schedule_, offsets_, counts_;
    std::vector<uint32_t> edges_, previous0_, previous1_, heads_;
    size_t batches_ = 0, samples_ = 0;
    double seconds_ = 0;
};
}

std::unique_ptr<BatchEvaluator> make_emulated_evaluator(const Encoding& e, const EvaluationOptions& o) {
    return std::unique_ptr<BatchEvaluator>(new EmulatedEvaluator(e, o));
}
} // namespace qbcs
