#ifndef ML_TRAIN_TRAINING_H
#define ML_TRAIN_TRAINING_H

#include "types.h"
#include "tensor.h"
#include "network.h"
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

namespace ml {

// ---------------------------------------------------------------------------
// Data Normalization
// ---------------------------------------------------------------------------

struct Normalizer {
    std::vector<NormParams> params;

    void fit(const std::vector<TelemetrySample>& data) {
        if (data.empty()) return;
        uint32_t dim = data[0].values.size();
        params.resize(dim);

        for (uint32_t j = 0; j < dim; ++j) {
            double sum = 0, sum2 = 0;
            int count = 0;
            for (const auto& s : data) {
                if (j < s.values.size() && (s.valid.empty() || s.valid[j])) {
                    sum += s.values[j];
                    sum2 += (double)s.values[j] * s.values[j];
                    count++;
                }
            }
            if (count > 0) {
                params[j].mean = (float)(sum / count);
                float var = (float)(sum2 / count - (double)params[j].mean * params[j].mean);
                params[j].stddev = std::sqrt(std::max(var, 1e-8f));
            }
        }
    }

    Tensor normalize(const TelemetrySample& sample) const {
        Tensor r(1, params.size());
        for (uint32_t j = 0; j < params.size(); ++j) {
            float val = (j < sample.values.size()) ? sample.values[j] : 0.0f;
            r[j] = (val - params[j].mean) / params[j].stddev;
        }
        return r;
    }

    Tensor normalizeBatch(const std::vector<TelemetrySample>& data,
                          uint32_t start, uint32_t count) const {
        Tensor batch(count, params.size());
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t idx = start + i;
            if (idx >= data.size()) break;
            for (uint32_t j = 0; j < params.size(); ++j) {
                float val = (j < data[idx].values.size()) ? data[idx].values[j] : 0.0f;
                batch(i, j) = (val - params[j].mean) / params[j].stddev;
            }
        }
        return batch;
    }
};

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

class Trainer {
public:
    // Train an autoencoder for anomaly detection
    TrainResult trainAutoencoder(
        Network& net,
        const std::vector<TelemetrySample>& data,
        const TrainConfig& cfg) {

        TrainResult result;
        if (data.empty()) return result;

        // Fit normalizer
        normalizer_.fit(data);

        // Split train/val
        uint32_t valSize = std::max(1u, (uint32_t)(data.size() * cfg.validationSplit));
        uint32_t trainSize = data.size() - valSize;

        // Build index shuffle
        std::vector<uint32_t> indices(trainSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(42);

        float bestValLoss = 1e30f;
        uint32_t patienceCounter = 0;

        for (uint32_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            float epochLoss = 0;
            uint32_t batches = 0;

            // Mini-batch training
            for (uint32_t b = 0; b < trainSize; b += cfg.batchSize) {
                uint32_t batchEnd = std::min(b + cfg.batchSize, trainSize);
                uint32_t batchLen = batchEnd - b;

                // Build batch
                Tensor batch(batchLen, net.inputDim());
                for (uint32_t i = 0; i < batchLen; ++i) {
                    auto row = normalizer_.normalize(data[indices[b + i]]);
                    batch.setRow(i, row);
                }

                // Forward (autoencoder: target = input)
                Tensor output = net.forward(batch,
                    cfg.quant == QuantMode::TERNARY ? QuantMode::TERNARY : QuantMode::FULL_PRECISION);

                // Backward (MSE loss)
                net.backward(batch);

                // Update
                net.update(cfg);

                epochLoss += batch.mse(output) * batchLen;
                batches++;
            }

            epochLoss /= trainSize;
            result.lossHistory.push_back(epochLoss);

            // Validation
            Tensor valBatch = normalizer_.normalizeBatch(data, trainSize, valSize);
            Tensor valOut = net.forward(valBatch, QuantMode::FULL_PRECISION);
            float valLoss = valBatch.mse(valOut);

            // Learning rate decay
            TrainConfig mutableCfg = cfg;
            mutableCfg.lr *= std::pow(cfg.lrDecay, (float)epoch);

            // Early stopping
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= cfg.patience) {
                    result.earlyStopped = true;
                    result.epochsTrained = epoch + 1;
                    break;
                }
            }

            result.epochsTrained = epoch + 1;
        }

        result.finalLoss = result.lossHistory.back();
        result.valLoss = bestValLoss;

        // Compute anomaly threshold (mean + 3*std of reconstruction errors on training data)
        if (cfg.anomalyThreshold == 0.0f) {
            std::vector<float> errors;
            for (uint32_t i = 0; i < trainSize; i += cfg.batchSize) {
                uint32_t batchLen = std::min(cfg.batchSize, trainSize - i);
                Tensor batch = normalizer_.normalizeBatch(data, i, batchLen);
                Tensor output = net.forward(batch, QuantMode::FULL_PRECISION);
                for (uint32_t r = 0; r < batchLen; ++r) {
                    float err = 0;
                    for (uint32_t c = 0; c < net.inputDim(); ++c) {
                        float d = batch(r, c) - output(r, c);
                        err += d * d;
                    }
                    errors.push_back(err / net.inputDim());
                }
            }
            float mean = 0, std = 0;
            for (auto e : errors) mean += e;
            mean /= errors.size();
            for (auto e : errors) std += (e - mean) * (e - mean);
            std = std::sqrt(std / errors.size());
            result.anomalyThreshold = mean + 3.0f * std;
        } else {
            result.anomalyThreshold = cfg.anomalyThreshold;
        }

        return result;
    }

    // Train a classifier
    TrainResult trainClassifier(
        Network& net,
        const std::vector<TelemetrySample>& data,
        const std::vector<uint32_t>& labels,
        uint32_t numClasses,
        const TrainConfig& cfg) {

        TrainResult result;
        if (data.empty()) return result;

        normalizer_.fit(data);

        uint32_t trainSize = data.size();
        std::vector<uint32_t> indices(trainSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(42);

        for (uint32_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            float epochLoss = 0;

            for (uint32_t b = 0; b < trainSize; b += cfg.batchSize) {
                uint32_t batchLen = std::min(cfg.batchSize, trainSize - b);

                Tensor batch(batchLen, net.inputDim());
                Tensor target(batchLen, numClasses, 0.0f);  // one-hot

                for (uint32_t i = 0; i < batchLen; ++i) {
                    auto row = normalizer_.normalize(data[indices[b + i]]);
                    batch.setRow(i, row);
                    target(i, labels[indices[b + i]]) = 1.0f;
                }

                Tensor output = net.forward(batch,
                    cfg.quant == QuantMode::TERNARY ? QuantMode::TERNARY : QuantMode::FULL_PRECISION);

                // Cross-entropy loss gradient (softmax output - target)
                net.backward(target);
                net.update(cfg);

                // Cross-entropy loss for logging
                for (uint32_t i = 0; i < batchLen; ++i) {
                    for (uint32_t c = 0; c < numClasses; ++c) {
                        if (target(i, c) > 0.5f)
                            epochLoss -= std::log(std::max(output(i, c), 1e-7f));
                    }
                }
            }

            epochLoss /= trainSize;
            result.lossHistory.push_back(epochLoss);
            result.epochsTrained = epoch + 1;
        }

        result.finalLoss = result.lossHistory.back();
        return result;
    }

    const Normalizer& normalizer() const { return normalizer_; }

private:
    Normalizer normalizer_;
};

}  // namespace ml

#endif  // ML_TRAIN_TRAINING_H
