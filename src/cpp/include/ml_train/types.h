#ifndef ML_TRAIN_TYPES_H
#define ML_TRAIN_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

namespace ml {

// ---------------------------------------------------------------------------
// Activation Functions
// ---------------------------------------------------------------------------

enum class Activation : uint8_t {
    NONE = 0,
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU,
    SOFTMAX
};

// ---------------------------------------------------------------------------
// Model Architecture Type
// ---------------------------------------------------------------------------

enum class ArchType : uint8_t {
    AUTOENCODER   = 0,   // Anomaly detection via reconstruction error
    CLASSIFIER    = 1,   // Fault classification
    PREDICTOR     = 2,   // Time-series prediction (sliding window)
};

// ---------------------------------------------------------------------------
// Quantization Mode
// ---------------------------------------------------------------------------

enum class QuantMode : uint8_t {
    FULL_PRECISION = 0,  // float32 weights
    TERNARY        = 1,  // {-1, 0, +1} with TWN threshold
    BINARY         = 2,  // {-1, +1} sign-based
};

// ---------------------------------------------------------------------------
// Ternary Weight Encoding (2 bits per weight)
// 00 = 0, 01 = +1, 10 = -1, 11 = reserved
// ---------------------------------------------------------------------------

static constexpr uint8_t TERNARY_ZERO = 0b00;
static constexpr uint8_t TERNARY_POS  = 0b01;
static constexpr uint8_t TERNARY_NEG  = 0b10;

// Pack 4 ternary weights per byte
inline uint8_t packTernary4(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    auto enc = [](int8_t w) -> uint8_t {
        if (w > 0) return TERNARY_POS;
        if (w < 0) return TERNARY_NEG;
        return TERNARY_ZERO;
    };
    return (enc(w0) << 6) | (enc(w1) << 4) | (enc(w2) << 2) | enc(w3);
}

inline void unpackTernary4(uint8_t packed, int8_t out[4]) {
    static const int8_t lut[4] = {0, 1, -1, 0};
    out[0] = lut[(packed >> 6) & 0x3];
    out[1] = lut[(packed >> 4) & 0x3];
    out[2] = lut[(packed >> 2) & 0x3];
    out[3] = lut[(packed >> 0) & 0x3];
}

// ---------------------------------------------------------------------------
// Layer Descriptor (for serialization)
// ---------------------------------------------------------------------------

struct LayerDesc {
    uint32_t   inputSize  = 0;
    uint32_t   outputSize = 0;
    Activation activation = Activation::RELU;
    bool       hasBias    = true;
};

// ---------------------------------------------------------------------------
// Model Header (binary format)
// Magic: "SDNM" (SDN Model), version 1
// ---------------------------------------------------------------------------

static constexpr uint32_t MODEL_MAGIC   = 0x4D4E4453;  // "SDNM" little-endian
static constexpr uint16_t MODEL_VERSION = 1;

struct ModelHeader {
    uint32_t magic      = MODEL_MAGIC;
    uint16_t version    = MODEL_VERSION;
    uint8_t  archType   = 0;         // ArchType enum
    uint8_t  quantMode  = 0;         // QuantMode enum
    uint32_t numLayers  = 0;
    uint32_t inputDim   = 0;
    uint32_t outputDim  = 0;
    float    trainLoss  = 0.0f;
    uint32_t epochs     = 0;
    // Normalization parameters (per input feature)
    uint32_t normCount  = 0;         // number of normalization entries
};

struct NormParams {
    float mean   = 0.0f;
    float stddev = 1.0f;
};

// ---------------------------------------------------------------------------
// Training Configuration
// ---------------------------------------------------------------------------

struct TrainConfig {
    ArchType  arch        = ArchType::AUTOENCODER;
    QuantMode quant       = QuantMode::TERNARY;
    float     lr          = 0.001f;   // learning rate
    float     lrDecay     = 0.999f;   // per-epoch decay
    float     momentum    = 0.9f;
    float     weightDecay = 1e-5f;
    uint32_t  epochs      = 100;
    uint32_t  batchSize   = 32;
    float     ternaryThresholdScale = 0.7f;  // TWN: threshold = scale * mean(|w|)
    float     anomalyThreshold = 0.0f;       // auto-computed if 0
    float     validationSplit  = 0.1f;
    uint32_t  patience         = 10;         // early stopping patience
    bool      verbose          = false;
};

// ---------------------------------------------------------------------------
// Training Result
// ---------------------------------------------------------------------------

struct TrainResult {
    float    finalLoss    = 0.0f;
    float    valLoss      = 0.0f;
    uint32_t epochsTrained = 0;
    bool     earlyStopped = false;
    float    anomalyThreshold = 0.0f;  // for autoencoders
    std::vector<float> lossHistory;
};

// ---------------------------------------------------------------------------
// XTCE Types (minimal)
// ---------------------------------------------------------------------------

enum class XTCEParamType : uint8_t {
    FLOAT32 = 0,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    UINT8,
    UINT16,
    UINT32,
    BOOLEAN,
    ENUMERATED,
    STRING
};

struct XTCEParameter {
    std::string name;
    std::string description;
    XTCEParamType type = XTCEParamType::FLOAT64;
    std::string units;

    // Alarm limits
    bool hasLimits = false;
    float warnLow  = 0, warnHigh  = 0;
    float critLow  = 0, critHigh  = 0;

    // Calibration (polynomial: y = c0 + c1*x + c2*x^2 + ...)
    std::vector<float> calibCoeffs;

    // For enumerated types
    std::vector<std::pair<int, std::string>> enumValues;
};

struct XTCEContainer {
    std::string name;
    std::string description;
    std::vector<std::string> parameterRefs;  // names of contained parameters
    float rateHz = 1.0f;                      // expected data rate
};

struct XTCETelemetryDef {
    std::string name;
    std::string version;
    std::vector<XTCEParameter> parameters;
    std::vector<XTCEContainer> containers;
};

// ---------------------------------------------------------------------------
// Telemetry Sample (a single timestep of all parameters)
// ---------------------------------------------------------------------------

struct TelemetrySample {
    double timestamp;                    // epoch seconds or MJD
    std::vector<float> values;           // one per parameter
    std::vector<bool>  valid;            // validity flags
};

// ---------------------------------------------------------------------------
// Anomaly Alert
// ---------------------------------------------------------------------------

enum class AlertSeverity : uint8_t {
    INFO = 0,
    WARNING,
    CRITICAL
};

struct AnomalyAlert {
    double        timestamp;
    AlertSeverity severity;
    float         score;           // reconstruction error or prediction residual
    float         threshold;
    std::string   description;
    std::vector<std::pair<int, float>> topContributors;  // (param_idx, contribution)
};

}  // namespace ml

#endif  // ML_TRAIN_TYPES_H
