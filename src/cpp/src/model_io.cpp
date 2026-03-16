#include "ml_train/model_io.h"
#include <cstring>

namespace ml {

// ---------------------------------------------------------------------------
// Binary Writer Helper
// ---------------------------------------------------------------------------

class BinaryWriter {
public:
    template<typename T>
    void write(T value) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
        data_.insert(data_.end(), ptr, ptr + sizeof(T));
    }

    void writeBytes(const uint8_t* ptr, size_t len) {
        data_.insert(data_.end(), ptr, ptr + len);
    }

    void writeFloat(float v) { write(v); }
    void writeU32(uint32_t v) { write(v); }
    void writeU16(uint16_t v) { write(v); }
    void writeU8(uint8_t v) { write(v); }

    std::vector<uint8_t> finish() { return std::move(data_); }

private:
    std::vector<uint8_t> data_;
};

// ---------------------------------------------------------------------------
// Binary Reader Helper
// ---------------------------------------------------------------------------

class BinaryReader {
public:
    BinaryReader(const std::vector<uint8_t>& data) : data_(data), pos_(0) {}

    template<typename T>
    T read() {
        T value;
        if (pos_ + sizeof(T) > data_.size()) return T{};
        std::memcpy(&value, data_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    float readFloat() { return read<float>(); }
    uint32_t readU32() { return read<uint32_t>(); }
    uint16_t readU16() { return read<uint16_t>(); }
    uint8_t readU8() { return read<uint8_t>(); }

    void readBytes(uint8_t* dst, size_t len) {
        if (pos_ + len > data_.size()) return;
        std::memcpy(dst, data_.data() + pos_, len);
        pos_ += len;
    }

    bool valid() const { return pos_ <= data_.size(); }
    size_t remaining() const { return data_.size() - pos_; }

private:
    const std::vector<uint8_t>& data_;
    size_t pos_;
};

// ---------------------------------------------------------------------------
// Serialize
// ---------------------------------------------------------------------------

std::vector<uint8_t> serializeModel(
    const Network& net,
    ArchType arch,
    QuantMode quant,
    const std::vector<NormParams>& normParams,
    float trainLoss,
    uint32_t epochs,
    float anomalyThreshold) {

    BinaryWriter w;

    // Header
    w.writeU32(MODEL_MAGIC);
    w.writeU16(MODEL_VERSION);
    w.writeU8(static_cast<uint8_t>(arch));
    w.writeU8(static_cast<uint8_t>(quant));
    w.writeU32(net.numLayers());
    w.writeU32(net.inputDim());
    w.writeU32(net.outputDim());
    w.writeFloat(trainLoss);
    w.writeU32(epochs);
    w.writeU32(normParams.size());

    // Anomaly threshold
    w.writeFloat(anomalyThreshold);

    // Normalization params
    for (const auto& np : normParams) {
        w.writeFloat(np.mean);
        w.writeFloat(np.stddev);
    }

    // Layers
    for (const auto& layer : net.layers()) {
        // Layer descriptor
        w.writeU32(layer.weights.rows());
        w.writeU32(layer.weights.cols());
        w.writeU8(static_cast<uint8_t>(layer.act));
        w.writeU8(1);  // hasBias

        if (quant == QuantMode::TERNARY) {
            // Pack ternary weights: 4 per byte
            // Also store per-layer scale factor
            float meanAbs = layer.weights.meanAbs();
            float threshold = 0.7f * meanAbs;
            float scaleSum = 0;
            int scaleCount = 0;

            // First pass: compute scale
            std::vector<int8_t> ternary(layer.weights.size());
            for (uint32_t i = 0; i < layer.weights.size(); ++i) {
                float val = layer.weights[i];
                if (val > threshold) {
                    ternary[i] = 1;
                    scaleSum += std::abs(val);
                    scaleCount++;
                } else if (val < -threshold) {
                    ternary[i] = -1;
                    scaleSum += std::abs(val);
                    scaleCount++;
                } else {
                    ternary[i] = 0;
                }
            }
            float scale = scaleCount > 0 ? scaleSum / scaleCount : 1.0f;
            w.writeFloat(scale);

            // Pack 4 weights per byte
            uint32_t numWeights = layer.weights.size();
            uint32_t packedSize = (numWeights + 3) / 4;
            for (uint32_t i = 0; i < packedSize; ++i) {
                int8_t w0 = (i * 4 + 0 < numWeights) ? ternary[i * 4 + 0] : 0;
                int8_t w1 = (i * 4 + 1 < numWeights) ? ternary[i * 4 + 1] : 0;
                int8_t w2 = (i * 4 + 2 < numWeights) ? ternary[i * 4 + 2] : 0;
                int8_t w3 = (i * 4 + 3 < numWeights) ? ternary[i * 4 + 3] : 0;
                w.writeU8(packTernary4(w0, w1, w2, w3));
            }
        } else {
            // Full precision float32
            for (uint32_t i = 0; i < layer.weights.size(); ++i) {
                w.writeFloat(layer.weights[i]);
            }
        }

        // Bias (always float32)
        for (uint32_t i = 0; i < layer.bias.size(); ++i) {
            w.writeFloat(layer.bias[i]);
        }
    }

    return w.finish();
}

// ---------------------------------------------------------------------------
// Deserialize
// ---------------------------------------------------------------------------

bool deserializeModel(
    const std::vector<uint8_t>& data,
    Network& net,
    ModelHeader& header,
    std::vector<NormParams>& normParams,
    float& anomalyThreshold) {

    BinaryReader r(data);

    // Header
    header.magic = r.readU32();
    if (header.magic != MODEL_MAGIC) return false;

    header.version = r.readU16();
    if (header.version != MODEL_VERSION) return false;

    header.archType = r.readU8();
    header.quantMode = r.readU8();
    header.numLayers = r.readU32();
    header.inputDim = r.readU32();
    header.outputDim = r.readU32();
    header.trainLoss = r.readFloat();
    header.epochs = r.readU32();
    header.normCount = r.readU32();

    anomalyThreshold = r.readFloat();

    // Normalization params
    normParams.resize(header.normCount);
    for (uint32_t i = 0; i < header.normCount; ++i) {
        normParams[i].mean = r.readFloat();
        normParams[i].stddev = r.readFloat();
    }

    // Layers
    std::vector<LayerDesc> descs;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> layerData;

    QuantMode quant = static_cast<QuantMode>(header.quantMode);

    for (uint32_t l = 0; l < header.numLayers; ++l) {
        LayerDesc desc;
        desc.inputSize = r.readU32();
        desc.outputSize = r.readU32();
        desc.activation = static_cast<Activation>(r.readU8());
        desc.hasBias = r.readU8() != 0;
        descs.push_back(desc);

        uint32_t numWeights = desc.inputSize * desc.outputSize;
        std::vector<float> weights(numWeights);

        if (quant == QuantMode::TERNARY) {
            float scale = r.readFloat();
            uint32_t packedSize = (numWeights + 3) / 4;
            for (uint32_t i = 0; i < packedSize; ++i) {
                uint8_t packed = r.readU8();
                int8_t vals[4];
                unpackTernary4(packed, vals);
                for (int k = 0; k < 4; ++k) {
                    uint32_t idx = i * 4 + k;
                    if (idx < numWeights)
                        weights[idx] = vals[k] * scale;
                }
            }
        } else {
            for (uint32_t i = 0; i < numWeights; ++i)
                weights[i] = r.readFloat();
        }

        std::vector<float> bias(desc.outputSize);
        for (uint32_t i = 0; i < desc.outputSize; ++i)
            bias[i] = r.readFloat();

        layerData.push_back({weights, bias});
    }

    // Build network and load weights
    net.build(descs);
    for (uint32_t l = 0; l < header.numLayers; ++l) {
        auto& layer = net.layers()[l];
        for (uint32_t i = 0; i < layer.weights.size(); ++i)
            layer.weights[i] = layerData[l].first[i];
        for (uint32_t i = 0; i < layer.bias.size(); ++i)
            layer.bias[i] = layerData[l].second[i];
    }

    return r.valid();
}

uint32_t estimateModelSize(
    const std::vector<LayerDesc>& layers,
    QuantMode quant) {

    uint32_t size = sizeof(ModelHeader) + sizeof(float);  // header + anomaly threshold

    for (const auto& l : layers) {
        size += sizeof(LayerDesc);
        uint32_t numWeights = l.inputSize * l.outputSize;

        if (quant == QuantMode::TERNARY) {
            size += sizeof(float);             // scale
            size += (numWeights + 3) / 4;      // packed ternary
        } else {
            size += numWeights * sizeof(float); // float32
        }

        size += l.outputSize * sizeof(float);  // bias
    }

    return size;
}

}  // namespace ml
