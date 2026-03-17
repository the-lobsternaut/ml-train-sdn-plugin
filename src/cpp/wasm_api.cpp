/**
 * ml-train-sdn-plugin WASM API
 *
 * Neural network creation, training (autoencoder/classifier/predictor),
 * model serialization, XTCE telemetry parsing, synthetic data generation.
 *
 * JSON-in / JSON-out pattern for complex structures.
 * Embind exports for direct JS/browser usage.
 */

#include "ml_train/network.h"
#include "ml_train/training.h"
#include "ml_train/model_io.h"
#include "ml_train/types.h"
#include "ml_train/xtce_parser.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
using namespace emscripten;
#endif

#include <string>
#include <cstring>

#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#define HAS_JSON 1
#else
#define HAS_JSON 0
#endif

using namespace ml;

// ============================================================================
// Version
// ============================================================================

static std::string version() { return "1.0.0"; }

// ============================================================================
// Persistent state (WASM is single-threaded)
// ============================================================================

static Network g_network;
static Trainer g_trainer;

// ============================================================================
// JSON API Wrappers
// ============================================================================

#if HAS_JSON

/// Build autoencoder network.
/// Input: {"inputDim":10, "hiddenSizes":[64,32,16], "seed":42}
static std::string buildAutoencoder_json(const std::string& input) {
    auto j = json::parse(input);
    uint32_t inputDim = j["inputDim"].get<uint32_t>();
    auto hiddenSizes = j["hiddenSizes"].get<std::vector<uint32_t>>();
    if (j.contains("seed")) g_network.setSeed(j["seed"].get<uint32_t>());

    g_network.buildAutoencoder(inputDim, hiddenSizes);

    return json({
        {"inputDim", g_network.inputDim()},
        {"outputDim", g_network.outputDim()},
        {"numLayers", g_network.numLayers()},
        {"status", "ok"}
    }).dump();
}

/// Build classifier network.
/// Input: {"inputDim":10, "hiddenSizes":[64,32], "numClasses":5, "seed":42}
static std::string buildClassifier_json(const std::string& input) {
    auto j = json::parse(input);
    uint32_t inputDim = j["inputDim"].get<uint32_t>();
    auto hiddenSizes = j["hiddenSizes"].get<std::vector<uint32_t>>();
    uint32_t numClasses = j["numClasses"].get<uint32_t>();
    if (j.contains("seed")) g_network.setSeed(j["seed"].get<uint32_t>());

    g_network.buildClassifier(inputDim, hiddenSizes, numClasses);

    return json({
        {"inputDim", g_network.inputDim()},
        {"outputDim", g_network.outputDim()},
        {"numLayers", g_network.numLayers()},
        {"status", "ok"}
    }).dump();
}

/// Build predictor network.
/// Input: {"windowSize":20, "hiddenSizes":[64,32], "outputDim":5, "seed":42}
static std::string buildPredictor_json(const std::string& input) {
    auto j = json::parse(input);
    uint32_t windowSize = j["windowSize"].get<uint32_t>();
    auto hiddenSizes = j["hiddenSizes"].get<std::vector<uint32_t>>();
    uint32_t outputDim = j["outputDim"].get<uint32_t>();
    if (j.contains("seed")) g_network.setSeed(j["seed"].get<uint32_t>());

    g_network.buildPredictor(windowSize, hiddenSizes, outputDim);

    return json({
        {"inputDim", g_network.inputDim()},
        {"outputDim", g_network.outputDim()},
        {"numLayers", g_network.numLayers()},
        {"status", "ok"}
    }).dump();
}

/// Train autoencoder on telemetry data.
/// Input: {"data":[{"timestamp":..., "values":[...]},...], "config":{epochs, lr, batchSize,...}}
static std::string trainAutoencoder_json(const std::string& input) {
    auto j = json::parse(input);

    // Parse telemetry samples
    std::vector<TelemetrySample> data;
    for (auto& s : j["data"]) {
        TelemetrySample sample;
        sample.timestamp = s.value("timestamp", 0.0);
        sample.values = s["values"].get<std::vector<float>>();
        data.push_back(sample);
    }

    // Parse training config
    TrainConfig cfg;
    cfg.arch = ArchType::AUTOENCODER;
    if (j.contains("config")) {
        auto& c = j["config"];
        cfg.epochs = c.value("epochs", 100u);
        cfg.lr = c.value("lr", 0.001f);
        cfg.batchSize = c.value("batchSize", 32u);
        cfg.momentum = c.value("momentum", 0.9f);
        cfg.weightDecay = c.value("weightDecay", 1e-5f);
        cfg.validationSplit = c.value("validationSplit", 0.1f);
        cfg.patience = c.value("patience", 10u);
        cfg.lrDecay = c.value("lrDecay", 0.999f);
        if (c.contains("quant"))
            cfg.quant = static_cast<QuantMode>(c["quant"].get<int>());
    }

    auto result = g_trainer.trainAutoencoder(g_network, data, cfg);

    json out;
    out["finalLoss"] = result.finalLoss;
    out["valLoss"] = result.valLoss;
    out["epochsTrained"] = result.epochsTrained;
    out["earlyStopped"] = result.earlyStopped;
    out["anomalyThreshold"] = result.anomalyThreshold;
    out["lossHistory"] = result.lossHistory;
    return out.dump();
}

/// Train classifier on labeled telemetry data.
/// Input: {"data":[...], "labels":[0,1,2,...], "numClasses":3, "config":{...}}
static std::string trainClassifier_json(const std::string& input) {
    auto j = json::parse(input);

    std::vector<TelemetrySample> data;
    for (auto& s : j["data"]) {
        TelemetrySample sample;
        sample.timestamp = s.value("timestamp", 0.0);
        sample.values = s["values"].get<std::vector<float>>();
        data.push_back(sample);
    }

    auto labels = j["labels"].get<std::vector<uint32_t>>();
    uint32_t numClasses = j["numClasses"].get<uint32_t>();

    TrainConfig cfg;
    cfg.arch = ArchType::CLASSIFIER;
    if (j.contains("config")) {
        auto& c = j["config"];
        cfg.epochs = c.value("epochs", 100u);
        cfg.lr = c.value("lr", 0.001f);
        cfg.batchSize = c.value("batchSize", 32u);
    }

    auto result = g_trainer.trainClassifier(g_network, data, labels, numClasses, cfg);

    return json({
        {"finalLoss", result.finalLoss},
        {"epochsTrained", result.epochsTrained},
        {"lossHistory", result.lossHistory}
    }).dump();
}

/// Serialize trained model to binary (.sdn-model).
/// Input: {"arch":0, "quant":1, "anomalyThreshold":0.5}
/// Returns base64-encoded model bytes (or writes to WASM memory).
static std::string serializeModel_json(const std::string& input) {
    auto j = json::parse(input);
    auto arch = static_cast<ArchType>(j.value("arch", 0));
    auto quant = static_cast<QuantMode>(j.value("quant", 1));
    float threshold = j.value("anomalyThreshold", 0.0f);

    auto& normalizer = g_trainer.normalizer();
    auto bytes = serializeModel(g_network, arch, quant, normalizer.params, 0.0f, 0, threshold);

    // Encode as base64 for JSON transport
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t n = (uint32_t)bytes[i] << 16;
        if (i + 1 < bytes.size()) n |= (uint32_t)bytes[i+1] << 8;
        if (i + 2 < bytes.size()) n |= bytes[i+2];
        encoded += b64[(n >> 18) & 0x3F];
        encoded += b64[(n >> 12) & 0x3F];
        encoded += (i + 1 < bytes.size()) ? b64[(n >> 6) & 0x3F] : '=';
        encoded += (i + 2 < bytes.size()) ? b64[n & 0x3F] : '=';
    }

    return json({
        {"modelBase64", encoded},
        {"sizeBytes", bytes.size()}
    }).dump();
}

/// Run forward pass on current network.
/// Input: {"values":[...]}
static std::string forward_json(const std::string& input) {
    auto j = json::parse(input);
    auto values = j["values"].get<std::vector<float>>();

    Tensor inputT(1, values.size());
    for (size_t i = 0; i < values.size(); i++) inputT[i] = values[i];

    auto output = g_network.forward(inputT);
    std::vector<float> result(output.size());
    for (uint32_t i = 0; i < output.size(); i++) result[i] = output[i];

    return json({{"output", result}}).dump();
}

/// Parse XTCE XML and return telemetry definition.
/// Input: {"xml":"<SpaceSystem>...</SpaceSystem>"}
static std::string parseXTCE_json(const std::string& input) {
    auto j = json::parse(input);
    auto xml = j["xml"].get<std::string>();
    auto def = parseXTCE(xml);

    json params = json::array();
    for (auto& p : def.parameters) {
        json pj;
        pj["name"] = p.name;
        pj["description"] = p.description;
        pj["type"] = static_cast<int>(p.type);
        pj["units"] = p.units;
        pj["hasLimits"] = p.hasLimits;
        if (p.hasLimits) {
            pj["warnLow"] = p.warnLow; pj["warnHigh"] = p.warnHigh;
            pj["critLow"] = p.critLow; pj["critHigh"] = p.critHigh;
        }
        params.push_back(pj);
    }

    json containers = json::array();
    for (auto& c : def.containers) {
        containers.push_back({
            {"name", c.name}, {"description", c.description},
            {"parameterRefs", c.parameterRefs}, {"rateHz", c.rateHz}
        });
    }

    return json({
        {"name", def.name}, {"version", def.version},
        {"parameters", params}, {"containers", containers}
    }).dump();
}

/// Generate synthetic telemetry.
/// Input: {"xml":"...", "numSamples":1000, "anomalyRate":0.05, "seed":42}
static std::string generateSyntheticTelemetry_json(const std::string& input) {
    auto j = json::parse(input);
    auto xml = j["xml"].get<std::string>();
    auto def = parseXTCE(xml);
    uint32_t numSamples = j.value("numSamples", 1000u);
    float anomalyRate = j.value("anomalyRate", 0.05f);
    uint32_t seed = j.value("seed", 42u);

    auto samples = generateSyntheticTelemetry(def, numSamples, anomalyRate, seed);

    json out = json::array();
    for (auto& s : samples) {
        out.push_back({{"timestamp", s.timestamp}, {"values", s.values}});
    }
    return out.dump();
}

/// Estimate model size for a given architecture.
/// Input: {"layers":[{"inputSize":10,"outputSize":64},{"inputSize":64,"outputSize":10}], "quant":1}
static std::string estimateModelSize_json(const std::string& input) {
    auto j = json::parse(input);

    std::vector<LayerDesc> layers;
    for (auto& lj : j["layers"]) {
        LayerDesc ld;
        ld.inputSize = lj["inputSize"].get<uint32_t>();
        ld.outputSize = lj["outputSize"].get<uint32_t>();
        layers.push_back(ld);
    }

    auto quant = static_cast<QuantMode>(j.value("quant", 1));
    uint32_t size = estimateModelSize(layers, quant);

    return json({{"sizeBytes", size}}).dump();
}

/// Get network info.
static std::string getNetworkInfo_json() {
    return json({
        {"inputDim", g_network.inputDim()},
        {"outputDim", g_network.outputDim()},
        {"numLayers", g_network.numLayers()}
    }).dump();
}

#endif // HAS_JSON

// ============================================================================
// SDN Plugin ABI
// ============================================================================

#ifdef __EMSCRIPTEN__

extern "C" {

EMSCRIPTEN_KEEPALIVE
void* sdn_malloc(size_t size) { return malloc(size); }

EMSCRIPTEN_KEEPALIVE
void sdn_free(void* ptr) { free(ptr); }

} // extern "C"

// ============================================================================
// Embind API
// ============================================================================

EMSCRIPTEN_BINDINGS(sdn_ml_train) {
    function("version", &version);

#if HAS_JSON
    // Network building
    function("buildAutoencoder", &buildAutoencoder_json);
    function("buildClassifier", &buildClassifier_json);
    function("buildPredictor", &buildPredictor_json);
    function("getNetworkInfo", &getNetworkInfo_json);

    // Training
    function("trainAutoencoder", &trainAutoencoder_json);
    function("trainClassifier", &trainClassifier_json);

    // Inference (forward pass)
    function("forward", &forward_json);

    // Model I/O
    function("serializeModel", &serializeModel_json);
    function("estimateModelSize", &estimateModelSize_json);

    // XTCE
    function("parseXTCE", &parseXTCE_json);
    function("generateSyntheticTelemetry", &generateSyntheticTelemetry_json);
#endif
}

#endif // __EMSCRIPTEN__
