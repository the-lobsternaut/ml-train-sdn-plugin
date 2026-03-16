#ifndef ML_TRAIN_NETWORK_H
#define ML_TRAIN_NETWORK_H

#include "types.h"
#include "tensor.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace ml {

// ---------------------------------------------------------------------------
// Dense Layer with Ternary Quantization Support
// ---------------------------------------------------------------------------

struct DenseLayer {
    Tensor weights;      // [input x output] full-precision (latent)
    Tensor bias;         // [1 x output]
    Activation act;

    // Cached forward pass values (for backprop)
    Tensor input;        // input to this layer
    Tensor preAct;       // before activation
    Tensor output;       // after activation

    // Gradient accumulators
    Tensor dWeights;
    Tensor dBias;

    // Momentum buffers
    Tensor vWeights;
    Tensor vBias;

    // Ternary quantized weights (used during forward pass in QAT mode)
    Tensor ternaryWeights;
    float  ternaryThreshold = 0.0f;
    float  ternaryScale     = 1.0f;

    void init(uint32_t inputSize, uint32_t outputSize, Activation activation,
              std::mt19937& rng) {
        act = activation;
        weights.resize(inputSize, outputSize);
        bias.resize(1, outputSize, 0.0f);
        dWeights.resize(inputSize, outputSize);
        dBias.resize(1, outputSize);
        vWeights.resize(inputSize, outputSize);
        vBias.resize(1, outputSize);

        // He init for ReLU, Xavier for others
        if (act == Activation::RELU || act == Activation::LEAKY_RELU)
            weights.heInit(rng);
        else
            weights.xavierInit(rng);
    }
};

// ---------------------------------------------------------------------------
// Activation Functions (forward + derivative)
// ---------------------------------------------------------------------------

namespace act_fn {

inline Tensor forward(const Tensor& x, Activation act) {
    switch (act) {
    case Activation::RELU: {
        Tensor r(x.rows(), x.cols());
        for (uint32_t i = 0; i < x.size(); ++i)
            r[i] = x[i] > 0 ? x[i] : 0;
        return r;
    }
    case Activation::LEAKY_RELU: {
        Tensor r(x.rows(), x.cols());
        for (uint32_t i = 0; i < x.size(); ++i)
            r[i] = x[i] > 0 ? x[i] : 0.01f * x[i];
        return r;
    }
    case Activation::SIGMOID: {
        Tensor r(x.rows(), x.cols());
        for (uint32_t i = 0; i < x.size(); ++i)
            r[i] = 1.0f / (1.0f + std::exp(-x[i]));
        return r;
    }
    case Activation::TANH: {
        Tensor r(x.rows(), x.cols());
        for (uint32_t i = 0; i < x.size(); ++i)
            r[i] = std::tanh(x[i]);
        return r;
    }
    case Activation::SOFTMAX: {
        Tensor r(x.rows(), x.cols());
        for (uint32_t row = 0; row < x.rows(); ++row) {
            float maxVal = -1e30f;
            for (uint32_t j = 0; j < x.cols(); ++j)
                maxVal = std::max(maxVal, x(row, j));
            float sumExp = 0;
            for (uint32_t j = 0; j < x.cols(); ++j) {
                r(row, j) = std::exp(x(row, j) - maxVal);
                sumExp += r(row, j);
            }
            for (uint32_t j = 0; j < x.cols(); ++j)
                r(row, j) /= sumExp;
        }
        return r;
    }
    default:
        return x;
    }
}

// Returns d(activation)/d(preActivation), element-wise
inline Tensor derivative(const Tensor& preAct, const Tensor& output, Activation act) {
    Tensor d(preAct.rows(), preAct.cols());
    switch (act) {
    case Activation::RELU:
        for (uint32_t i = 0; i < preAct.size(); ++i)
            d[i] = preAct[i] > 0 ? 1.0f : 0.0f;
        break;
    case Activation::LEAKY_RELU:
        for (uint32_t i = 0; i < preAct.size(); ++i)
            d[i] = preAct[i] > 0 ? 1.0f : 0.01f;
        break;
    case Activation::SIGMOID:
        for (uint32_t i = 0; i < output.size(); ++i)
            d[i] = output[i] * (1.0f - output[i]);
        break;
    case Activation::TANH:
        for (uint32_t i = 0; i < output.size(); ++i)
            d[i] = 1.0f - output[i] * output[i];
        break;
    case Activation::SOFTMAX:
        // For softmax + cross-entropy, derivative is handled at loss level
        d.fill(1.0f);
        break;
    default:
        d.fill(1.0f);
        break;
    }
    return d;
}

}  // namespace act_fn

// ---------------------------------------------------------------------------
// Neural Network
// ---------------------------------------------------------------------------

class Network {
public:
    Network() = default;

    void setSeed(uint32_t seed) { rng_.seed(seed); }

    // Build from layer descriptors
    void build(const std::vector<LayerDesc>& descs) {
        layers_.clear();
        for (const auto& d : descs) {
            DenseLayer layer;
            layer.init(d.inputSize, d.outputSize, d.activation, rng_);
            layers_.push_back(std::move(layer));
        }
    }

    // Build autoencoder: input → encoder → bottleneck → decoder → input
    void buildAutoencoder(uint32_t inputDim, const std::vector<uint32_t>& hiddenSizes) {
        std::vector<LayerDesc> descs;

        // Encoder
        uint32_t prevSize = inputDim;
        for (auto sz : hiddenSizes) {
            descs.push_back({prevSize, sz, Activation::RELU, true});
            prevSize = sz;
        }

        // Decoder (mirror)
        for (int i = (int)hiddenSizes.size() - 2; i >= 0; --i) {
            descs.push_back({prevSize, hiddenSizes[i], Activation::RELU, true});
            prevSize = hiddenSizes[i];
        }

        // Output layer (linear, to match input range)
        descs.push_back({prevSize, inputDim, Activation::NONE, true});

        build(descs);
    }

    // Build classifier: input → hidden... → numClasses (softmax)
    void buildClassifier(uint32_t inputDim, const std::vector<uint32_t>& hiddenSizes,
                         uint32_t numClasses) {
        std::vector<LayerDesc> descs;
        uint32_t prevSize = inputDim;
        for (auto sz : hiddenSizes) {
            descs.push_back({prevSize, sz, Activation::RELU, true});
            prevSize = sz;
        }
        descs.push_back({prevSize, numClasses, Activation::SOFTMAX, true});
        build(descs);
    }

    // Build predictor: input (window of past values) → hidden... → predicted values
    void buildPredictor(uint32_t windowSize, const std::vector<uint32_t>& hiddenSizes,
                        uint32_t outputDim) {
        std::vector<LayerDesc> descs;
        uint32_t prevSize = windowSize;
        for (auto sz : hiddenSizes) {
            descs.push_back({prevSize, sz, Activation::RELU, true});
            prevSize = sz;
        }
        descs.push_back({prevSize, outputDim, Activation::NONE, true});
        build(descs);
    }

    // ===== Forward Pass =====
    Tensor forward(const Tensor& input, QuantMode quant = QuantMode::FULL_PRECISION) {
        Tensor x = input;
        for (auto& layer : layers_) {
            layer.input = x;

            // Use ternary weights during QAT or inference
            const Tensor& w = (quant == QuantMode::TERNARY)
                ? quantizeTernary(layer) : layer.weights;

            layer.preAct = x.matmul(w);
            layer.preAct.addBias(layer.bias);
            layer.output = act_fn::forward(layer.preAct, layer.act);
            x = layer.output;
        }
        return x;
    }

    // ===== Backward Pass =====
    // Computes gradients w.r.t. MSE loss: L = (1/N) * sum((output - target)^2)
    void backward(const Tensor& target) {
        // Output layer gradient: dL/dOutput = 2/N * (output - target)
        Tensor& lastOut = layers_.back().output;
        Tensor dOut = (lastOut - target) * (2.0f / target.size());

        for (int i = (int)layers_.size() - 1; i >= 0; --i) {
            auto& layer = layers_[i];

            // Activation derivative
            Tensor dAct = act_fn::derivative(layer.preAct, layer.output, layer.act);
            Tensor delta = dOut * dAct;

            // Weight gradients: dW = input^T * delta
            layer.dWeights = layer.input.T().matmul(delta);
            layer.dBias = delta.sumRows();

            // Propagate gradient to previous layer
            if (i > 0) {
                dOut = delta.matmul(layer.weights.T());
            }
        }
    }

    // ===== SGD Update with Momentum =====
    void update(const TrainConfig& cfg) {
        for (auto& layer : layers_) {
            // Momentum: v = momentum * v - lr * grad
            layer.vWeights = layer.vWeights * cfg.momentum - layer.dWeights * cfg.lr;
            layer.vBias    = layer.vBias * cfg.momentum - layer.dBias * cfg.lr;

            // Weight decay
            if (cfg.weightDecay > 0) {
                layer.weights -= layer.weights * cfg.weightDecay;
            }

            // Update
            layer.weights += layer.vWeights;
            layer.bias    += layer.vBias;

            // Zero gradients
            layer.dWeights.zero();
            layer.dBias.zero();
        }
    }

    // ===== Ternary Quantization (TWN) =====
    // threshold = scale * mean(|W|)
    // w_t = +1 if w > threshold, -1 if w < -threshold, 0 otherwise
    // scale factor alpha = mean(|w| for w where |w| > threshold)
    const Tensor& quantizeTernary(DenseLayer& layer, float thresholdScale = 0.7f) {
        float meanAbs = layer.weights.meanAbs();
        float threshold = thresholdScale * meanAbs;

        layer.ternaryWeights.resize(layer.weights.rows(), layer.weights.cols());
        layer.ternaryThreshold = threshold;

        float scaleSum = 0;
        int scaleCount = 0;

        for (uint32_t i = 0; i < layer.weights.size(); ++i) {
            float w = layer.weights[i];
            if (w > threshold) {
                layer.ternaryWeights[i] = 1.0f;
                scaleSum += std::abs(w);
                scaleCount++;
            } else if (w < -threshold) {
                layer.ternaryWeights[i] = -1.0f;
                scaleSum += std::abs(w);
                scaleCount++;
            } else {
                layer.ternaryWeights[i] = 0.0f;
            }
        }

        layer.ternaryScale = scaleCount > 0 ? scaleSum / scaleCount : 1.0f;

        // Scale the ternary weights by alpha for better approximation
        for (uint32_t i = 0; i < layer.ternaryWeights.size(); ++i)
            layer.ternaryWeights[i] *= layer.ternaryScale;

        return layer.ternaryWeights;
    }

    // Accessors
    std::vector<DenseLayer>& layers() { return layers_; }
    const std::vector<DenseLayer>& layers() const { return layers_; }
    uint32_t inputDim() const { return layers_.empty() ? 0 : layers_[0].weights.rows(); }
    uint32_t outputDim() const { return layers_.empty() ? 0 : layers_.back().weights.cols(); }
    uint32_t numLayers() const { return layers_.size(); }

private:
    std::vector<DenseLayer> layers_;
    std::mt19937 rng_{42};
};

}  // namespace ml

#endif  // ML_TRAIN_NETWORK_H
