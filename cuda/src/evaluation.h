#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace qbcs {

struct Encoding;
struct Sample;

struct EvaluationOptions {
    std::string backend = "cuda";
    int device = 0;
    int batch_capacity = 128;
    std::size_t memory_mb = 0;
    int wait_us = 200;
    bool warp_decode = false;
};

class BatchEvaluator {
public:
    virtual ~BatchEvaluator() = default;
    virtual int capacity() const = 0;
    virtual std::string description() const = 0;
    virtual void evaluate(const std::vector<const Sample*>& samples,
                          std::vector<int>& fitness) = 0;
    virtual std::string statistics() const = 0;
};

std::unique_ptr<BatchEvaluator> make_cuda_evaluator(const Encoding&, const EvaluationOptions&);
std::unique_ptr<BatchEvaluator> make_emulated_evaluator(const Encoding&, const EvaluationOptions&);

// One owning worker merges independent calls across trials. Callers block until every
// requested sample is complete, so their sample buffers and algorithm dependencies stay
// unchanged. The queue holds borrowed pointers, never extra copies of large populations.
class EvaluationService {
public:
    explicit EvaluationService(std::unique_ptr<BatchEvaluator> backend,
                               int wait_us = 200, std::size_t max_pending_requests = 256);
    ~EvaluationService();
    EvaluationService(const EvaluationService&) = delete;
    EvaluationService& operator=(const EvaluationService&) = delete;

    void evaluate(const Sample* samples, std::size_t count, int* fitness);
    int capacity() const;
    std::string description() const;
    std::string statistics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qbcs
