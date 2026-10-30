#include "evaluation.h"
#include "qbcs.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace qbcs {

struct EvaluationService::Impl {
    struct Request {
        const Sample* samples;
        std::size_t count;
        int* fitness;
        std::size_t scheduled = 0;
        std::size_t completed = 0;
        bool done = false;
        std::exception_ptr error;
        std::condition_variable ready;
        Request(const Sample* s, std::size_t n, int* f) : samples(s), count(n), fitness(f) {}
    };
    struct Part {
        std::shared_ptr<Request> request;
        std::size_t begin;
        std::size_t count;
    };

    std::unique_ptr<BatchEvaluator> backend;
    const int batch_capacity;
    const int wait_us;
    const std::size_t max_pending;
    mutable std::mutex mutex;
    std::condition_variable available, space;
    std::deque<std::shared_ptr<Request>> queue;
    std::thread worker;
    bool stopping = false;
    std::exception_ptr failure;
    std::size_t batches = 0, evaluated = 0, largest_batch = 0;

    Impl(std::unique_ptr<BatchEvaluator> b, int wait, std::size_t pending)
        : backend(std::move(b)), batch_capacity(backend ? backend->capacity() : 0),
          wait_us(wait), max_pending(pending) {
        if (!backend || batch_capacity <= 0 || wait_us < 0 || max_pending == 0)
            throw std::invalid_argument("invalid evaluation service configuration");
        worker = std::thread([this] { run(); });
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        available.notify_all();
        space.notify_all();
        if (worker.joinable()) worker.join();
    }

    std::size_t waiting_samples() const {
        std::size_t n = 0;
        for (const auto& request : queue) {
            n += std::min(request->count - request->scheduled,
                          static_cast<std::size_t>(batch_capacity) - n);
            if (n >= static_cast<std::size_t>(batch_capacity)) break;
        }
        return n;
    }

    void fail_all(std::exception_ptr error, const std::vector<Part>& active) {
        std::lock_guard<std::mutex> lock(mutex);
        failure = error;
        stopping = true;
        for (const auto& part : active) {
            part.request->error = error;
            part.request->done = true;
            part.request->ready.notify_all();
        }
        for (const auto& request : queue) {
            request->error = error;
            request->done = true;
            request->ready.notify_all();
        }
        queue.clear();
        space.notify_all();
        available.notify_all();
    }

    void run() noexcept {
        std::vector<Part> active;
        try {
            std::vector<const Sample*> samples;
            std::vector<int> fitness;
            samples.reserve(batch_capacity);
            fitness.reserve(batch_capacity);
            active.reserve(batch_capacity);
            for (;;) {
                samples.clear();
                active.clear();
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    available.wait(lock, [&] { return stopping || !queue.empty(); });
                    if (stopping && queue.empty()) return;
                    if (wait_us && waiting_samples() < static_cast<std::size_t>(batch_capacity)) {
                        const auto deadline = std::chrono::steady_clock::now()
                                            + std::chrono::microseconds(wait_us);
                        available.wait_until(lock, deadline, [&] {
                            return stopping || waiting_samples() >= static_cast<std::size_t>(batch_capacity);
                        });
                    }
                    while (!queue.empty() && samples.size() < static_cast<std::size_t>(batch_capacity)) {
                        auto request = queue.front();
                        const std::size_t take = std::min(request->count - request->scheduled,
                            static_cast<std::size_t>(batch_capacity) - samples.size());
                        const std::size_t begin = request->scheduled;
                        active.push_back(Part{request, begin, take});
                        for (std::size_t i = 0; i < take; ++i)
                            samples.push_back(request->samples + begin + i);
                        request->scheduled += take;
                        if (request->scheduled == request->count) queue.pop_front();
                    }
                    space.notify_all();
                }
                fitness.clear();
                backend->evaluate(samples, fitness);
                if (fitness.size() != samples.size())
                    throw std::runtime_error("evaluation backend returned the wrong result count");
                std::size_t offset = 0;
                for (const auto& part : active) {
                    std::copy_n(fitness.data() + offset, part.count,
                                part.request->fitness + part.begin);
                    offset += part.count;
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    ++batches;
                    evaluated += samples.size();
                    largest_batch = std::max(largest_batch, samples.size());
                    for (const auto& part : active) {
                        part.request->completed += part.count;
                        if (part.request->completed == part.request->count) {
                            part.request->done = true;
                            part.request->ready.notify_one();
                        }
                    }
                }
            }
        } catch (...) {
            fail_all(std::current_exception(), active);
        }
    }
};

EvaluationService::EvaluationService(std::unique_ptr<BatchEvaluator> backend,
                                     int wait_us, std::size_t max_pending_requests)
    : impl_(std::make_unique<Impl>(std::move(backend), wait_us, max_pending_requests)) {}

EvaluationService::~EvaluationService() = default;

void EvaluationService::evaluate(const Sample* samples, std::size_t count, int* fitness) {
    if (!count) return;
    if (!samples || !fitness) throw std::invalid_argument("null evaluation buffers");
    auto request = std::make_shared<Impl::Request>(samples, count, fitness);
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->space.wait(lock, [&] {
        return impl_->stopping || impl_->failure || impl_->queue.size() < impl_->max_pending;
    });
    if (impl_->failure) std::rethrow_exception(impl_->failure);
    if (impl_->stopping) throw std::runtime_error("evaluation service is stopped");
    impl_->queue.push_back(request);
    impl_->available.notify_one();
    request->ready.wait(lock, [&] { return request->done; });
    if (request->error) std::rethrow_exception(request->error);
}

int EvaluationService::capacity() const { return impl_->batch_capacity; }
std::string EvaluationService::description() const { return impl_->backend->description(); }
std::string EvaluationService::statistics() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::ostringstream out;
    out << "Batches: " << impl_->batches << "; samples: " << impl_->evaluated
        << "; mean batch: " << (impl_->batches ? double(impl_->evaluated) / impl_->batches : 0.0)
        << "; largest batch: " << impl_->largest_batch;
    const std::string backend_stats = impl_->backend->statistics();
    if (!backend_stats.empty()) out << "; " << backend_stats;
    return out.str();
}

}  // namespace qbcs
