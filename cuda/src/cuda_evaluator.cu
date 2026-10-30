// CUDA evaluator for the unchanged stepwise synthesis search.
// All GPU arithmetic that influences fitness is integer arithmetic.
#include "evaluation.h"
#include "batch_layout.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qbcs {
namespace {

void checked(cudaError_t result, const char* expression) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(result));
}
#define CUDA_CHECK(call) checked((call), #call)

template<class T> class DeviceArray {
public:
    T* data = nullptr;
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if (data) cudaFree(data); }
    void allocate(size_t n) { if (n) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T))); }
};
template<class T> class PinnedArray {
public:
    T* data = nullptr;
    PinnedArray() = default;
    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
    ~PinnedArray() { if (data) cudaFreeHost(data); }
    void allocate(size_t n) { CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), std::max<size_t>(1, n) * sizeof(T))); }
};
struct Stream {
    cudaStream_t value = nullptr;
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    ~Event() { if (value) cudaEventDestroy(value); }
};

__global__ void decode_thread_kernel(const gpu_core::Step* steps, const uint8_t* input,
    size_t input_bytes, int l3_size, const int* schedule, const int* offsets,
    int active_steps, int batch, uint32_t* edges) {
    const size_t total = size_t(active_steps) * batch;
    for (size_t slot = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         slot < total; slot += size_t(blockDim.x) * gridDim.x) {
        const int sample = int(slot % batch);
        const uint8_t* row = input + size_t(sample) * input_bytes;
        gpu_core::decode_step(steps[schedule[slot]], row, row + l3_size,
            edges + size_t(offsets[slot]) * batch + sample, size_t(batch));
    }
}

// Optional alternative to benchmark on the actual GPU. All lanes participate in
// every shuffle/ballot. The stable rank is the original MSB-first index on ties.
// This kernel changes the execution structure only; it uses the same repaired L4.
__global__ void decode_warp_kernel(const gpu_core::Step* steps, const uint8_t* input,
    size_t input_bytes, int l3_size, const int* schedule, const int* offsets,
    int active_steps, int batch, uint32_t* edges) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = int(threadIdx.x) & 31;
    size_t warp = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const size_t warp_stride = (size_t(blockDim.x) * gridDim.x) >> 5;
    const size_t total = size_t(active_steps) * batch;
    for (size_t slot = warp; slot < total; slot += warp_stride) {
        const int sample = int(slot % batch);
        const gpu_core::Step s = steps[schedule[slot]];
        const int d = s.distance;
        uint32_t* out = edges + size_t(offsets[slot]) * batch + sample;
        if (s.node_bits == 0) {
            if (lane == 0) out[0] = gpu_core::edge(s.start, gpu_core::low_bit(s.diff));
            continue;
        }
        const uint8_t* row = input + size_t(sample) * input_bytes;
        const uint8_t* sequence = row + l3_size + s.l4_offset;
        int priority = -1, position = 0;
        if (lane < d) {
            priority = 0;
            for (int b = 0; b < s.node_bits; ++b)
                priority = (priority << 1) | row[s.l3_offset + lane * s.node_bits + b];
            int k = 0;
            for (int p = gpu_core::max_state_bits - 1; p >= 0; --p)
                if ((s.diff >> p) & 1u) { if (k == lane) position = p; ++k; }
        }
        int rank = 0;
        for (int j = 0; j < d; ++j) {
            const int other = __shfl_sync(mask, priority, j);
            if (other > priority || (other == priority && j < lane)) ++rank;
        }
        uint32_t route = s.start;
        for (int j = 0; j < d; ++j) {
            const int order = __shfl_sync(mask, rank, j);
            const int pos = __shfl_sync(mask, position, j);
            if (order < lane) route ^= uint32_t(1) << pos;
        }
        const bool first_zero = lane < d - 1 && sequence[lane] == 0;
        const bool second_zero = lane < d - 1 && sequence[d - 1 + lane] == 0;
        const unsigned first = __ballot_sync(mask, first_zero);
        const unsigned second = __ballot_sync(mask, second_zero);
        const int middle_head = __popc(first);
        for (int base = 0; base < 2 * d - 1; base += 32) {
            const int emission = base + lane;
            int a = 0, b = 1;
            if (emission < d - 1) {
                const int zeros = __popc(first & ((uint32_t(1) << emission) - 1u));
                if (sequence[emission] == 0) { a = zeros; b = a + 1; }
                else { b = d - (emission - zeros); a = b - 1; }
            } else if (emission == d - 1) {
                a = middle_head; b = a + 1;
            } else if (emission < 2 * d - 1) {
                const int j = emission - d;
                const int zeros = __popc(second & ((uint32_t(1) << j) - 1u));
                if (sequence[d - 1 + j] == 0) { a = middle_head + 1 + zeros; b = a + 1; }
                else { b = middle_head - (j - zeros); a = b - 1; }
            }
            const uint32_t va = __shfl_sync(mask, route, a);
            const uint32_t vb = __shfl_sync(mask, route, b);
            if (emission < 2 * d - 1)
                out[size_t(emission) * batch] = gpu_core::edge(va, gpu_core::low_bit(va ^ vb));
        }
    }
}

__global__ void reduce_kernel(const uint32_t* edges, const int* counts, int batch,
    uint32_t* heads, uint32_t* previous0, uint32_t* previous1, int* fitness) {
    const int sample = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (sample < batch)
        fitness[sample] = gpu_core::reduce(edges + sample, counts[sample], size_t(batch),
            heads + sample, previous0 + sample, previous1 + sample);
}

class CudaEvaluator final : public BatchEvaluator {
public:
    CudaEvaluator(const Encoding& e, const EvaluationOptions& options)
        : layout_(e), options_(options) {
        if (options.device < 0 || options.batch_capacity < 1 ||
            options.memory_mb > std::numeric_limits<size_t>::max() / (1024 * 1024))
            throw std::invalid_argument("invalid CUDA evaluator capacity/device/memory budget");
        CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
        name_ = prop.name;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_.value, cudaStreamNonBlocking));
        for (auto& event : events_) CUDA_CHECK(cudaEventCreate(&event.value));
        size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const size_t budget = options.memory_mb ? std::min(free, options.memory_mb * size_t(1024 * 1024))
                                              : free / 5 * 4;
        const size_t meta = layout_.metadata_bytes();
        if (budget <= meta) throw std::runtime_error("GPU memory budget cannot hold topology metadata");
        capacity_ = int(std::min<size_t>(options.batch_capacity, (budget - meta) / layout_.bytes_per_sample()));
        if (capacity_ < 1) {
            std::ostringstream s;
            s << "GPU budget cannot hold one sample. Need at least " << layout_.bytes_per_sample() + meta
              << " device bytes; lower problem size or increase --gpu-memory-mb.";
            throw std::runtime_error(s.str());
        }
        const size_t b = size_t(capacity_);
        steps_.allocate(layout_.steps.size());
        input_.allocate(std::max<size_t>(1, layout_.input_bytes * b));
        schedule_.allocate(size_t(layout_.active_steps) * b);
        offsets_.allocate(size_t(layout_.active_steps) * b);
        counts_.allocate(b); fitness_.allocate(b);
        edges_.allocate(size_t(layout_.max_gates) * b);
        previous0_.allocate(size_t(layout_.max_gates) * b);
        previous1_.allocate(size_t(layout_.max_gates) * b);
        heads_.allocate(layout_.vertices * b);
        h_input_.allocate(layout_.input_bytes * b);
        h_schedule_.allocate(size_t(layout_.active_steps) * b);
        h_offsets_.allocate(size_t(layout_.active_steps) * b);
        h_counts_.allocate(b); h_fitness_.allocate(b);
        if (!layout_.steps.empty())
            CUDA_CHECK(cudaMemcpyAsync(steps_.data, layout_.steps.data(), meta, cudaMemcpyHostToDevice, stream_.value));
        CUDA_CHECK(cudaStreamSynchronize(stream_.value));
    }
    ~CudaEvaluator() override {
        // Members are then freed in this device context. Ignore destructor errors.
        cudaSetDevice(options_.device);
        if (stream_.value) cudaStreamSynchronize(stream_.value);
    }
    int capacity() const override { return capacity_; }
    std::string description() const override {
        std::ostringstream s;
        s << "CUDA device " << options_.device << " (" << name_ << "), "
          << (options_.warp_decode ? "warp" : "thread") << " decoder, capacity=" << capacity_
          << ", device bytes/sample=" << layout_.bytes_per_sample()
          << ", allocated device MiB=" << (layout_.metadata_bytes() + size_t(capacity_) * layout_.bytes_per_sample()) / 1048576.0;
        return s.str();
    }
    void evaluate(const std::vector<const Sample*>& samples, std::vector<int>& fitness) override {
        if (samples.size() > size_t(capacity_)) throw std::runtime_error("CUDA batch exceeds allocated capacity");
        const int batch = int(samples.size());
        fitness.resize(batch);
        if (!batch) return;
        if (!layout_.active_steps) {
            std::fill(fitness.begin(), fitness.end(), 0);
            ++batches_; samples_ += size_t(batch);
            minimum_batch_ = std::min(minimum_batch_, batch);
            maximum_batch_ = std::max(maximum_batch_, batch);
            return;
        }
        // CUDA's current device is per host thread; the service owns the stream,
        // but construction and execution may occur on different CPU threads.
        CUDA_CHECK(cudaSetDevice(options_.device));
        const auto t0 = std::chrono::steady_clock::now();
        layout_.pack(samples, h_input_.data, h_schedule_.data, h_offsets_.data, h_counts_.data);
        pack_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const cudaStream_t stream = stream_.value;
        CUDA_CHECK(cudaEventRecord(events_[0].value, stream));
        if (layout_.input_bytes)
            CUDA_CHECK(cudaMemcpyAsync(input_.data, h_input_.data, layout_.input_bytes * batch, cudaMemcpyHostToDevice, stream));
        const size_t schedule_bytes = size_t(layout_.active_steps) * batch * sizeof(int);
        if (schedule_bytes) {
            CUDA_CHECK(cudaMemcpyAsync(schedule_.data, h_schedule_.data, schedule_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(offsets_.data, h_offsets_.data, schedule_bytes, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(counts_.data, h_counts_.data, size_t(batch) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(heads_.data, 0, layout_.vertices * batch * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaEventRecord(events_[1].value, stream));
        if (layout_.active_steps) {
            const size_t tasks = size_t(layout_.active_steps) * batch;
            if (options_.warp_decode) {
                const int blocks = int(std::min<size_t>(65535, (tasks + 3) / 4));
                decode_warp_kernel<<<blocks, 128, 0, stream>>>(steps_.data, input_.data,
                    layout_.input_bytes, layout_.encoding.l3_size, schedule_.data, offsets_.data,
                    layout_.active_steps, batch, edges_.data);
            } else {
                const int blocks = int(std::min<size_t>(65535, (tasks + 127) / 128));
                decode_thread_kernel<<<blocks, 128, 0, stream>>>(steps_.data, input_.data,
                    layout_.input_bytes, layout_.encoding.l3_size, schedule_.data, offsets_.data,
                    layout_.active_steps, batch, edges_.data);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(events_[2].value, stream));
        // One warp per block distributes even modest batches over more SMs.
        // Lanes remain independent samples, preserving coalescing where possible.
        reduce_kernel<<<unsigned((size_t(batch) + 31) / 32), 32, 0, stream>>>(edges_.data, counts_.data, batch,
            heads_.data, previous0_.data, previous1_.data, fitness_.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(events_[3].value, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_fitness_.data, fitness_.data, size_t(batch) * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(events_[4].value, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < batch; ++i) fitness[i] = h_fitness_.data[i];
        for (int stage = 0; stage < 4; ++stage) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, events_[stage].value, events_[stage + 1].value));
            stage_ms_[stage] += ms;
        }
        ++batches_; samples_ += size_t(batch);
        minimum_batch_ = std::min(minimum_batch_, batch); maximum_batch_ = std::max(maximum_batch_, batch);
    }
    std::string statistics() const override {
        std::ostringstream s;
        s << "CUDA batches=" << batches_ << ", samples=" << samples_
          << ", batch min/mean/max=" << (batches_ ? minimum_batch_ : 0) << "/"
          << (batches_ ? double(samples_) / batches_ : 0.0) << "/" << maximum_batch_
          << "; CPU pack seconds=" << pack_seconds_
          << "; device stage ms H2D+clear/decode/reduce/D2H="
          << stage_ms_[0] << "/" << stage_ms_[1] << "/" << stage_ms_[2] << "/" << stage_ms_[3];
        return s.str();
    }
private:
    BatchLayout layout_;
    EvaluationOptions options_;
    int capacity_ = 1;
    std::string name_;
    Stream stream_;
    Event events_[5];
    DeviceArray<gpu_core::Step> steps_;
    DeviceArray<uint8_t> input_;
    DeviceArray<int> schedule_, offsets_, counts_, fitness_;
    DeviceArray<uint32_t> edges_, previous0_, previous1_, heads_;
    PinnedArray<uint8_t> h_input_;
    PinnedArray<int> h_schedule_, h_offsets_, h_counts_, h_fitness_;
    size_t batches_ = 0, samples_ = 0;
    int minimum_batch_ = 2147483647, maximum_batch_ = 0;
    double pack_seconds_ = 0, stage_ms_[4] = {};
};

} // namespace
std::unique_ptr<BatchEvaluator> make_cuda_evaluator(const Encoding& e, const EvaluationOptions& o) {
    return std::unique_ptr<BatchEvaluator>(new CudaEvaluator(e, o));
}
} // namespace qbcs
