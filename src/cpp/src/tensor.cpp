/**
 * Tensor — Compilation unit and extended operations
 *
 * The core Tensor class is defined inline in tensor.h for performance.
 * This file provides additional utility functions and ensures the
 * translation unit is compiled for link-time availability.
 *
 * Matrix operations follow standard numerical linear algebra conventions.
 * Reference: Golub & Van Loan, "Matrix Computations" (2013), 4th ed.
 */

#include "ml_train/tensor.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace ml {

// ── Tensor Utilities (non-inline, not on hot path) ──

/**
 * Create a tensor from a flat initializer list (row-major).
 * Useful for constructing small test tensors.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param values Flat array of values in row-major order
 * @return Constructed tensor
 */
Tensor tensor_from_data(uint32_t rows, uint32_t cols, const float* values) {
    Tensor t(rows, cols);
    std::memcpy(t.data(), values, rows * cols * sizeof(float));
    return t;
}

/**
 * Create an identity matrix.
 *
 * @param n Dimension (n x n)
 * @return Identity tensor
 */
Tensor tensor_eye(uint32_t n) {
    Tensor t(n, n, 0.0f);
    for (uint32_t i = 0; i < n; ++i)
        t(i, i) = 1.0f;
    return t;
}

/**
 * Concatenate tensors vertically (stack rows).
 * All tensors must have the same number of columns.
 *
 * @param tensors Vector of tensors to concatenate
 * @return Vertically stacked tensor
 */
Tensor tensor_vstack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) return Tensor();
    uint32_t cols = tensors[0].cols();
    uint32_t total_rows = 0;
    for (const auto& t : tensors) {
        total_rows += t.rows();
    }
    Tensor result(total_rows, cols);
    uint32_t offset = 0;
    for (const auto& t : tensors) {
        std::memcpy(result.data() + offset * cols, t.data(),
                    t.rows() * cols * sizeof(float));
        offset += t.rows();
    }
    return result;
}

/**
 * Compute L2 (Frobenius) norm of a tensor.
 *
 * @param t Input tensor
 * @return Frobenius norm
 */
float tensor_norm(const Tensor& t) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < t.size(); ++i)
        sum += t[i] * t[i];
    return std::sqrt(sum);
}

/**
 * Compute max absolute value in a tensor (infinity norm).
 *
 * @param t Input tensor
 * @return Max absolute value
 */
float tensor_max_abs(const Tensor& t) {
    float maxVal = 0.0f;
    for (uint32_t i = 0; i < t.size(); ++i)
        maxVal = std::max(maxVal, std::abs(t[i]));
    return maxVal;
}

/**
 * Pretty-print a tensor for debugging.
 *
 * @param t Input tensor
 * @param name Optional label
 * @return Formatted string representation
 */
std::string tensor_to_string(const Tensor& t, const std::string& name) {
    std::ostringstream oss;
    if (!name.empty()) oss << name << " ";
    oss << "[" << t.rows() << " x " << t.cols() << "]:\n";
    for (uint32_t i = 0; i < t.rows(); ++i) {
        oss << "  [";
        for (uint32_t j = 0; j < t.cols(); ++j) {
            if (j > 0) oss << ", ";
            oss << std::fixed << std::setprecision(6) << t(i, j);
        }
        oss << "]\n";
    }
    return oss.str();
}

/**
 * Compute element-wise absolute value.
 *
 * @param t Input tensor
 * @return Tensor of absolute values
 */
Tensor tensor_abs(const Tensor& t) {
    Tensor result(t.rows(), t.cols());
    for (uint32_t i = 0; i < t.size(); ++i)
        result[i] = std::abs(t[i]);
    return result;
}

/**
 * Clip tensor values to [lo, hi] range, returning a new tensor.
 *
 * @param t Input tensor
 * @param lo Lower bound
 * @param hi Upper bound
 * @return Clipped tensor
 */
Tensor tensor_clip(const Tensor& t, float lo, float hi) {
    Tensor result(t.rows(), t.cols());
    for (uint32_t i = 0; i < t.size(); ++i)
        result[i] = std::max(lo, std::min(hi, t[i]));
    return result;
}

/**
 * Compute argmax along columns for each row.
 * Returns a vector of column indices.
 *
 * @param t Input tensor
 * @return Vector of argmax indices (one per row)
 */
std::vector<uint32_t> tensor_argmax_rows(const Tensor& t) {
    std::vector<uint32_t> result(t.rows());
    for (uint32_t i = 0; i < t.rows(); ++i) {
        float maxVal = t(i, 0);
        uint32_t maxIdx = 0;
        for (uint32_t j = 1; j < t.cols(); ++j) {
            if (t(i, j) > maxVal) {
                maxVal = t(i, j);
                maxIdx = j;
            }
        }
        result[i] = maxIdx;
    }
    return result;
}

}  // namespace ml
