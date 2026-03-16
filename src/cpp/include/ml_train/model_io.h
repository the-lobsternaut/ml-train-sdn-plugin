#ifndef ML_TRAIN_MODEL_IO_H
#define ML_TRAIN_MODEL_IO_H

#include "types.h"
#include "tensor.h"
#include "network.h"
#include <vector>
#include <cstdint>

namespace ml {

// ---------------------------------------------------------------------------
// Model Serialization
//
// Binary format (.sdn-model):
//   [ModelHeader]
//   [NormParams × normCount]
//   For each layer:
//     [LayerDesc]
//     [weight data] — packed ternary (2 bits/weight) or float32
//     [bias data]   — float32
//     [ternaryScale] — float32 (if ternary)
//
// Ternary packing: 4 weights per byte, row-major order.
// Padding to byte boundary at end of each layer's weights.
// ---------------------------------------------------------------------------

/// Serialize a trained network to compact binary format
std::vector<uint8_t> serializeModel(
    const Network& net,
    ArchType arch,
    QuantMode quant,
    const std::vector<NormParams>& normParams,
    float trainLoss,
    uint32_t epochs,
    float anomalyThreshold = 0.0f);

/// Deserialize a model from binary format
/// Returns false if format is invalid
bool deserializeModel(
    const std::vector<uint8_t>& data,
    Network& net,
    ModelHeader& header,
    std::vector<NormParams>& normParams,
    float& anomalyThreshold);

/// Get model size in bytes (for a given architecture)
uint32_t estimateModelSize(
    const std::vector<LayerDesc>& layers,
    QuantMode quant);

}  // namespace ml

#endif  // ML_TRAIN_MODEL_IO_H
