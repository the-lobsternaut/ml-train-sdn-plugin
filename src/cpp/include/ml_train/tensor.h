#ifndef ML_TRAIN_TENSOR_H
#define ML_TRAIN_TENSOR_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace ml {

// ---------------------------------------------------------------------------
// Simple 2D Tensor (row-major)
// No heap gymnastics — just a flat vector with shape metadata.
// Enough for small networks (8-128 neurons per layer).
// ---------------------------------------------------------------------------

class Tensor {
public:
    Tensor() = default;
    Tensor(uint32_t rows, uint32_t cols, float fill = 0.0f)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    // Shape
    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t size() const { return rows_ * cols_; }

    // Element access
    float& operator()(uint32_t r, uint32_t c) { return data_[r * cols_ + c]; }
    float  operator()(uint32_t r, uint32_t c) const { return data_[r * cols_ + c]; }

    // Flat access
    float& operator[](uint32_t i) { return data_[i]; }
    float  operator[](uint32_t i) const { return data_[i]; }

    float*       data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Resize (destroys content)
    void resize(uint32_t rows, uint32_t cols, float fill = 0.0f) {
        rows_ = rows; cols_ = cols;
        data_.assign(rows * cols, fill);
    }

    // Fill
    void fill(float val) { std::fill(data_.begin(), data_.end(), val); }
    void zero() { fill(0.0f); }

    // Initialize with Xavier/Glorot uniform
    void xavierInit(std::mt19937& rng) {
        float limit = std::sqrt(6.0f / (rows_ + cols_));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& v : data_) v = dist(rng);
    }

    // Initialize with Kaiming/He normal
    void heInit(std::mt19937& rng) {
        float stddev = std::sqrt(2.0f / rows_);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& v : data_) v = dist(rng);
    }

    // ===== Operations =====

    // Matrix multiply: C = A * B  (this = A, B given, result returned)
    // A: [M x K], B: [K x N] → C: [M x N]
    Tensor matmul(const Tensor& B) const {
        assert(cols_ == B.rows_);
        Tensor C(rows_, B.cols_);
        for (uint32_t i = 0; i < rows_; ++i) {
            for (uint32_t k = 0; k < cols_; ++k) {
                float a_ik = data_[i * cols_ + k];
                if (a_ik == 0.0f) continue;
                for (uint32_t j = 0; j < B.cols_; ++j) {
                    C.data_[i * B.cols_ + j] += a_ik * B.data_[k * B.cols_ + j];
                }
            }
        }
        return C;
    }

    // Transpose
    Tensor T() const {
        Tensor result(cols_, rows_);
        for (uint32_t i = 0; i < rows_; ++i)
            for (uint32_t j = 0; j < cols_; ++j)
                result.data_[j * rows_ + i] = data_[i * cols_ + j];
        return result;
    }

    // Element-wise operations
    Tensor operator+(const Tensor& o) const {
        assert(size() == o.size());
        Tensor r(rows_, cols_);
        for (uint32_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }

    Tensor operator-(const Tensor& o) const {
        assert(size() == o.size());
        Tensor r(rows_, cols_);
        for (uint32_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }

    Tensor operator*(const Tensor& o) const {  // Hadamard
        assert(size() == o.size());
        Tensor r(rows_, cols_);
        for (uint32_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * o.data_[i];
        return r;
    }

    Tensor operator*(float s) const {
        Tensor r(rows_, cols_);
        for (uint32_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * s;
        return r;
    }

    // In-place add (for gradient accumulation)
    Tensor& operator+=(const Tensor& o) {
        assert(size() == o.size());
        for (uint32_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i];
        return *this;
    }

    Tensor& operator-=(const Tensor& o) {
        assert(size() == o.size());
        for (uint32_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i];
        return *this;
    }

    // Add bias vector (row vector) to each row
    void addBias(const Tensor& bias) {
        assert(bias.cols_ == cols_ && bias.rows_ == 1);
        for (uint32_t i = 0; i < rows_; ++i)
            for (uint32_t j = 0; j < cols_; ++j)
                data_[i * cols_ + j] += bias.data_[j];
    }

    // Sum along rows (result: 1 x cols)
    Tensor sumRows() const {
        Tensor r(1, cols_);
        for (uint32_t i = 0; i < rows_; ++i)
            for (uint32_t j = 0; j < cols_; ++j)
                r.data_[j] += data_[i * cols_ + j];
        return r;
    }

    // Row slice (returns a view-copy of row i as 1 x cols)
    Tensor row(uint32_t i) const {
        Tensor r(1, cols_);
        std::copy(data_.begin() + i * cols_, data_.begin() + (i + 1) * cols_, r.data_.begin());
        return r;
    }

    // Set row
    void setRow(uint32_t i, const Tensor& src) {
        assert(src.cols_ == cols_);
        std::copy(src.data_.begin(), src.data_.begin() + cols_, data_.begin() + i * cols_);
    }

    // Stats
    float sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0.0f);
    }

    float mean() const { return sum() / data_.size(); }

    float meanAbs() const {
        float s = 0;
        for (auto v : data_) s += std::abs(v);
        return s / data_.size();
    }

    float mse(const Tensor& target) const {
        assert(size() == target.size());
        float s = 0;
        for (uint32_t i = 0; i < data_.size(); ++i) {
            float d = data_[i] - target.data_[i];
            s += d * d;
        }
        return s / data_.size();
    }

    // Apply function element-wise
    Tensor apply(float (*fn)(float)) const {
        Tensor r(rows_, cols_);
        for (uint32_t i = 0; i < data_.size(); ++i) r.data_[i] = fn(data_[i]);
        return r;
    }

    // Clamp
    void clamp(float lo, float hi) {
        for (auto& v : data_) v = std::max(lo, std::min(hi, v));
    }

private:
    uint32_t rows_ = 0, cols_ = 0;
    std::vector<float> data_;
};

}  // namespace ml

#endif  // ML_TRAIN_TENSOR_H
