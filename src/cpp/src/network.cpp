/**
 * Neural Network — Compilation unit and Adam optimizer
 *
 * The core Network class (forward, backward, SGD update) is defined inline
 * in network.h for performance. This file provides:
 *   - Adam optimizer (adaptive learning rate)
 *   - Network diagnostic utilities
 *   - Gradient checking for verification
 *
 * References:
 *   - Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015), arXiv:1412.6980
 *   - Ruder, "An overview of gradient descent optimization algorithms" (2017), arXiv:1609.04747
 *   - LeCun et al., "Efficient BackProp" (1998), Neural Networks: Tricks of the Trade
 */

#include "ml_train/network.h"
#include "ml_train/training.h"
#include <cmath>
#include <cassert>
#include <algorithm>

namespace ml {

// ── Adam Optimizer State ──

struct AdamState {
    // First moment estimates (mean of gradients)
    std::vector<Tensor> m_weights;
    std::vector<Tensor> m_bias;
    // Second moment estimates (mean of squared gradients)
    std::vector<Tensor> v_weights;
    std::vector<Tensor> v_bias;
    uint32_t t = 0;  // timestep

    void init(const Network& net) {
        m_weights.clear();
        m_bias.clear();
        v_weights.clear();
        v_bias.clear();
        for (const auto& layer : net.layers()) {
            m_weights.emplace_back(layer.weights.rows(), layer.weights.cols(), 0.0f);
            m_bias.emplace_back(1, layer.bias.cols(), 0.0f);
            v_weights.emplace_back(layer.weights.rows(), layer.weights.cols(), 0.0f);
            v_bias.emplace_back(1, layer.bias.cols(), 0.0f);
        }
        t = 0;
    }
};

/**
 * Apply Adam optimizer update to network weights.
 *
 * Adam combines momentum (first moment) with RMSprop (second moment)
 * for adaptive per-parameter learning rates.
 *
 * Update rule (Kingma & Ba, 2015):
 *   m_t = β₁ m_{t-1} + (1 - β₁) g_t
 *   v_t = β₂ v_{t-1} + (1 - β₂) g_t²
 *   m̂_t = m_t / (1 - β₁^t)        (bias correction)
 *   v̂_t = v_t / (1 - β₂^t)        (bias correction)
 *   θ_t = θ_{t-1} - α m̂_t / (√v̂_t + ε)
 *
 * @param net          Network to update
 * @param state        Adam optimizer state (persistent across steps)
 * @param lr           Learning rate (α, typically 0.001)
 * @param beta1        First moment decay rate (typically 0.9)
 * @param beta2        Second moment decay rate (typically 0.999)
 * @param epsilon      Numerical stability constant (typically 1e-8)
 * @param weight_decay L2 regularization coefficient
 */
void adam_update(Network& net, AdamState& state,
                 float lr, float beta1, float beta2,
                 float epsilon, float weight_decay) {
    state.t++;
    float bc1 = 1.0f - std::pow(beta1, (float)state.t);  // bias correction 1
    float bc2 = 1.0f - std::pow(beta2, (float)state.t);  // bias correction 2

    auto& layers = net.layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];

        // Update first and second moments for weights
        for (uint32_t j = 0; j < layer.weights.size(); ++j) {
            float g = layer.dWeights[j];

            // L2 weight decay (decoupled, AdamW style)
            if (weight_decay > 0)
                layer.weights[j] -= lr * weight_decay * layer.weights[j];

            state.m_weights[i][j] = beta1 * state.m_weights[i][j] + (1.0f - beta1) * g;
            state.v_weights[i][j] = beta2 * state.v_weights[i][j] + (1.0f - beta2) * g * g;

            float m_hat = state.m_weights[i][j] / bc1;
            float v_hat = state.v_weights[i][j] / bc2;

            layer.weights[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        // Update first and second moments for biases
        for (uint32_t j = 0; j < layer.bias.size(); ++j) {
            float g = layer.dBias[j];

            state.m_bias[i][j] = beta1 * state.m_bias[i][j] + (1.0f - beta1) * g;
            state.v_bias[i][j] = beta2 * state.v_bias[i][j] + (1.0f - beta2) * g * g;

            float m_hat = state.m_bias[i][j] / bc1;
            float v_hat = state.v_bias[i][j] / bc2;

            layer.bias[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        // Zero gradients
        layer.dWeights.zero();
        layer.dBias.zero();
    }
}

/**
 * Numerical gradient check using central differences.
 *
 * Verifies analytical gradients by computing:
 *   ∂L/∂w ≈ (L(w + ε) - L(w - ε)) / (2ε)
 *
 * Returns the maximum relative error across all weights.
 * Should be < 1e-5 for correct implementations.
 *
 * Reference: CS231n, "Gradient checking and advanced optimization"
 *
 * @param net    Network (will be modified temporarily)
 * @param input  Input tensor
 * @param target Target tensor
 * @param eps    Perturbation size (default 1e-5)
 * @return Maximum relative error
 */
float gradient_check(Network& net, const Tensor& input, const Tensor& target,
                     float eps) {
    // Compute analytical gradients
    net.forward(input);
    net.backward(target);

    float max_rel_error = 0.0f;

    auto& layers = net.layers();
    for (size_t l = 0; l < layers.size(); ++l) {
        auto& layer = layers[l];

        // Check weight gradients
        for (uint32_t i = 0; i < layer.weights.size(); ++i) {
            float analytical = layer.dWeights[i];

            // Forward pass with w + eps
            float orig = layer.weights[i];
            layer.weights[i] = orig + eps;
            Tensor out_plus = net.forward(input);
            float loss_plus = out_plus.mse(target);

            // Forward pass with w - eps
            layer.weights[i] = orig - eps;
            Tensor out_minus = net.forward(input);
            float loss_minus = out_minus.mse(target);

            // Restore
            layer.weights[i] = orig;

            // Central difference
            float numerical = (loss_plus - loss_minus) / (2.0f * eps);

            // Relative error
            float denom = std::max(std::abs(analytical) + std::abs(numerical), 1e-8f);
            float rel_error = std::abs(analytical - numerical) / denom;
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }

    // Recompute analytical gradients (were corrupted by perturbations)
    net.forward(input);
    net.backward(target);

    return max_rel_error;
}

/**
 * Count total trainable parameters in the network.
 *
 * @param net Network
 * @return Total number of weight + bias parameters
 */
uint32_t count_parameters(const Network& net) {
    uint32_t total = 0;
    for (const auto& layer : net.layers()) {
        total += layer.weights.size() + layer.bias.size();
    }
    return total;
}

/**
 * Compute sparsity ratio of ternary-quantized weights.
 * Returns fraction of weights that are zero after quantization.
 *
 * @param net Network (must have been quantized)
 * @return Sparsity ratio [0, 1]
 */
float ternary_sparsity(const Network& net) {
    uint32_t total = 0;
    uint32_t zeros = 0;
    for (const auto& layer : net.layers()) {
        if (layer.ternaryWeights.size() == 0) continue;
        for (uint32_t i = 0; i < layer.ternaryWeights.size(); ++i) {
            total++;
            if (std::abs(layer.ternaryWeights[i]) < 1e-10f)
                zeros++;
        }
    }
    return total > 0 ? (float)zeros / total : 0.0f;
}

}  // namespace ml
