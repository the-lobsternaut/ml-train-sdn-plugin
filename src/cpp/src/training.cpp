/**
 * Training — Compilation unit and additional training utilities
 *
 * The core Trainer class (trainAutoencoder, trainClassifier) is defined
 * inline in training.h. This file provides:
 *   - Time-series predictor training
 *   - Sliding window dataset construction
 *   - Cross-validation utilities
 *   - Training metrics computation
 *
 * References:
 *   - Goodfellow, Bengio, Courville, "Deep Learning" (2016), Ch. 8
 *   - Chollet, "Deep Learning with Python" (2017), Ch. 6 (time series)
 */

#include "ml_train/training.h"
#include <cassert>
#include <numeric>

namespace ml {

// ── Sliding Window Dataset ──

/**
 * Build a sliding-window dataset from time-series telemetry.
 * Each sample is a window of `window_size` consecutive timesteps
 * used to predict the next `horizon` timesteps.
 *
 * @param data         Input telemetry samples (time-ordered)
 * @param normalizer   Fitted normalizer
 * @param window_size  Number of past timesteps per input sample
 * @param horizon      Number of future timesteps to predict
 * @param inputs       Output: [N x (window_size * dim)] input tensor
 * @param targets      Output: [N x (horizon * dim)] target tensor
 */
void build_sliding_window(const std::vector<TelemetrySample>& data,
                          const Normalizer& normalizer,
                          uint32_t window_size, uint32_t horizon,
                          Tensor& inputs, Tensor& targets) {
    if (data.size() < window_size + horizon) return;

    uint32_t dim = normalizer.params.size();
    uint32_t n_samples = data.size() - window_size - horizon + 1;
    uint32_t input_dim = window_size * dim;
    uint32_t output_dim = horizon * dim;

    inputs.resize(n_samples, input_dim);
    targets.resize(n_samples, output_dim);

    for (uint32_t i = 0; i < n_samples; ++i) {
        // Fill input window
        for (uint32_t w = 0; w < window_size; ++w) {
            Tensor row = normalizer.normalize(data[i + w]);
            for (uint32_t d = 0; d < dim; ++d) {
                inputs(i, w * dim + d) = row[d];
            }
        }
        // Fill target horizon
        for (uint32_t h = 0; h < horizon; ++h) {
            Tensor row = normalizer.normalize(data[i + window_size + h]);
            for (uint32_t d = 0; d < dim; ++d) {
                targets(i, h * dim + d) = row[d];
            }
        }
    }
}

/**
 * Train a time-series predictor using sliding windows.
 *
 * The network takes a flattened window of past normalized values as input
 * and predicts the next `horizon` normalized values.
 *
 * @param net         Network (should be built with buildPredictor)
 * @param data        Time-ordered telemetry samples
 * @param window_size Number of past timesteps per input
 * @param horizon     Number of future timesteps to predict
 * @param cfg         Training configuration
 * @return Training result with loss history
 */
TrainResult train_predictor(Network& net,
                            const std::vector<TelemetrySample>& data,
                            uint32_t window_size, uint32_t horizon,
                            const TrainConfig& cfg) {
    TrainResult result;
    if (data.size() < window_size + horizon + 1) return result;

    // Fit normalizer
    Normalizer normalizer;
    normalizer.fit(data);

    // Build dataset
    Tensor inputs, targets;
    build_sliding_window(data, normalizer, window_size, horizon, inputs, targets);

    uint32_t n_samples = inputs.rows();
    if (n_samples == 0) return result;

    // Split train/val
    uint32_t val_size = std::max(1u, (uint32_t)(n_samples * cfg.validationSplit));
    uint32_t train_size = n_samples - val_size;

    // Shuffle indices (but preserve temporal order within windows)
    std::vector<uint32_t> indices(train_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    float best_val_loss = 1e30f;
    uint32_t patience_counter = 0;

    for (uint32_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0;

        for (uint32_t b = 0; b < train_size; b += cfg.batchSize) {
            uint32_t batch_len = std::min(cfg.batchSize, train_size - b);

            // Build batch from shuffled indices
            Tensor batch_in(batch_len, inputs.cols());
            Tensor batch_tgt(batch_len, targets.cols());
            for (uint32_t i = 0; i < batch_len; ++i) {
                uint32_t idx = indices[b + i];
                for (uint32_t j = 0; j < inputs.cols(); ++j)
                    batch_in(i, j) = inputs(idx, j);
                for (uint32_t j = 0; j < targets.cols(); ++j)
                    batch_tgt(i, j) = targets(idx, j);
            }

            Tensor output = net.forward(batch_in);
            net.backward(batch_tgt);
            net.update(cfg);

            epoch_loss += batch_in.mse(output) * batch_len;
        }

        epoch_loss /= train_size;
        result.lossHistory.push_back(epoch_loss);

        // Validation
        Tensor val_in(val_size, inputs.cols());
        Tensor val_tgt(val_size, targets.cols());
        for (uint32_t i = 0; i < val_size; ++i) {
            uint32_t idx = train_size + i;
            for (uint32_t j = 0; j < inputs.cols(); ++j)
                val_in(i, j) = inputs(idx, j);
            for (uint32_t j = 0; j < targets.cols(); ++j)
                val_tgt(i, j) = targets(idx, j);
        }
        Tensor val_out = net.forward(val_in);
        float val_loss = val_tgt.mse(val_out);

        // Early stopping
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= cfg.patience) {
                result.earlyStopped = true;
                result.epochsTrained = epoch + 1;
                break;
            }
        }

        result.epochsTrained = epoch + 1;
    }

    result.finalLoss = result.lossHistory.back();
    result.valLoss = best_val_loss;
    return result;
}

/**
 * Compute classification accuracy.
 *
 * @param net       Trained network
 * @param data      Input samples
 * @param labels    Ground truth labels
 * @param norm      Fitted normalizer
 * @return Accuracy [0, 1]
 */
float compute_accuracy(Network& net,
                       const std::vector<TelemetrySample>& data,
                       const std::vector<uint32_t>& labels,
                       const Normalizer& norm) {
    if (data.empty()) return 0.0f;

    uint32_t correct = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        Tensor input = norm.normalize(data[i]);
        Tensor output = net.forward(input);

        // Find predicted class (argmax)
        uint32_t pred = 0;
        float max_val = output[0];
        for (uint32_t j = 1; j < output.cols(); ++j) {
            if (output[j] > max_val) {
                max_val = output[j];
                pred = j;
            }
        }

        if (pred == labels[i]) correct++;
    }

    return (float)correct / data.size();
}

/**
 * Compute per-parameter anomaly contribution scores.
 * For autoencoders, this measures which input parameters contribute most
 * to the reconstruction error (useful for root-cause analysis).
 *
 * @param net       Trained autoencoder
 * @param sample    Input sample
 * @param norm      Fitted normalizer
 * @return Vector of (parameter_index, contribution_score) pairs,
 *         sorted by descending contribution
 */
std::vector<std::pair<int, float>> anomaly_contributions(
    Network& net,
    const TelemetrySample& sample,
    const Normalizer& norm) {

    Tensor input = norm.normalize(sample);
    Tensor output = net.forward(input);

    std::vector<std::pair<int, float>> contributions;
    float total_error = 0.0f;

    for (uint32_t j = 0; j < input.cols(); ++j) {
        float diff = input[j] - output[j];
        float err = diff * diff;
        contributions.push_back({(int)j, err});
        total_error += err;
    }

    // Normalize to fractions
    if (total_error > 1e-10f) {
        for (auto& c : contributions)
            c.second /= total_error;
    }

    // Sort by descending contribution
    std::sort(contributions.begin(), contributions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return contributions;
}

}  // namespace ml
