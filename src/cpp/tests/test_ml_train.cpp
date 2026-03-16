#include "ml_train/types.h"
#include "ml_train/tensor.h"
#include "ml_train/network.h"
#include "ml_train/training.h"
#include "ml_train/xtce_parser.h"
#include "ml_train/model_io.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

void assertNear(float a, float b, float tol, const char* msg = "") {
    if (std::abs(a - b) > tol) {
        std::cerr << "FAIL: " << msg << " expected " << b << " got " << a
                  << " (diff=" << std::abs(a - b) << ")\n";
        assert(false);
    }
}

// ===== Tensor Tests =====

void testTensorBasics() {
    ml::Tensor a(2, 3, 1.0f);
    assert(a.rows() == 2 && a.cols() == 3);
    assert(a(0, 0) == 1.0f);

    ml::Tensor b(2, 3, 2.0f);
    auto c = a + b;
    assert(c(0, 0) == 3.0f);

    auto d = a * b;
    assert(d(1, 2) == 2.0f);

    assertNear(a.sum(), 6.0f, 1e-5f, "sum");
    assertNear(a.mean(), 1.0f, 1e-5f, "mean");

    std::cout << "  Tensor basics ✓\n";
}

void testMatmul() {
    ml::Tensor a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    ml::Tensor b(3, 2);
    b(0, 0) = 7;  b(0, 1) = 8;
    b(1, 0) = 9;  b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;

    auto c = a.matmul(b);
    assert(c.rows() == 2 && c.cols() == 2);
    assertNear(c(0, 0), 58.0f, 1e-5f, "matmul[0,0]");   // 1*7 + 2*9 + 3*11
    assertNear(c(0, 1), 64.0f, 1e-5f, "matmul[0,1]");   // 1*8 + 2*10 + 3*12
    assertNear(c(1, 0), 139.0f, 1e-5f, "matmul[1,0]");  // 4*7 + 5*9 + 6*11
    assertNear(c(1, 1), 154.0f, 1e-5f, "matmul[1,1]");  // 4*8 + 5*10 + 6*12

    std::cout << "  Matmul ✓\n";
}

void testTranspose() {
    ml::Tensor a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    auto t = a.T();
    assert(t.rows() == 3 && t.cols() == 2);
    assertNear(t(0, 0), 1.0f, 1e-5f, "T[0,0]");
    assertNear(t(2, 1), 6.0f, 1e-5f, "T[2,1]");

    std::cout << "  Transpose ✓\n";
}

// ===== Ternary Encoding Tests =====

void testTernaryPacking() {
    uint8_t packed = ml::packTernary4(1, -1, 0, 1);
    int8_t out[4];
    ml::unpackTernary4(packed, out);
    assert(out[0] == 1);
    assert(out[1] == -1);
    assert(out[2] == 0);
    assert(out[3] == 1);

    packed = ml::packTernary4(0, 0, 0, 0);
    ml::unpackTernary4(packed, out);
    for (int i = 0; i < 4; ++i) assert(out[i] == 0);

    packed = ml::packTernary4(-1, -1, -1, -1);
    ml::unpackTernary4(packed, out);
    for (int i = 0; i < 4; ++i) assert(out[i] == -1);

    std::cout << "  Ternary packing ✓\n";
}

// ===== Network Forward Pass =====

void testForwardPass() {
    ml::Network net;
    net.setSeed(42);

    // Simple 4 → 8 → 4 autoencoder
    net.buildAutoencoder(4, {8, 2});

    // Create input
    ml::Tensor input(1, 4);
    input[0] = 0.5f; input[1] = -0.3f; input[2] = 0.8f; input[3] = -0.1f;

    auto output = net.forward(input);
    assert(output.rows() == 1 && output.cols() == 4);

    // Output should be different from input (untrained)
    float diff = input.mse(output);
    assert(diff > 0);

    std::cout << "  Forward pass: MSE=" << diff << " ✓\n";
}

// ===== Autoencoder Training =====

void testAutoencoderTraining() {
    ml::Network net;
    net.setSeed(42);
    net.buildAutoencoder(4, {16, 4});

    // Generate simple training data: sin/cos patterns
    std::vector<ml::TelemetrySample> data;
    for (int i = 0; i < 500; ++i) {
        ml::TelemetrySample sample;
        sample.timestamp = i;
        float t = i * 0.1f;
        sample.values = {
            std::sin(t) * 10.0f + 50.0f,       // oscillating ~50
            std::cos(t) * 5.0f + 25.0f,        // oscillating ~25
            20.0f + std::sin(t * 0.5f) * 2.0f,  // slow oscillation ~20
            100.0f + std::cos(t * 0.3f) * 8.0f   // oscillating ~100
        };
        data.push_back(sample);
    }

    ml::TrainConfig cfg;
    cfg.arch = ml::ArchType::AUTOENCODER;
    cfg.quant = ml::QuantMode::FULL_PRECISION;
    cfg.lr = 0.001f;
    cfg.epochs = 50;
    cfg.batchSize = 32;
    cfg.patience = 20;
    cfg.validationSplit = 0.1f;

    ml::Trainer trainer;
    auto result = trainer.trainAutoencoder(net, data, cfg);

    std::cout << "  Autoencoder: epochs=" << result.epochsTrained
              << " loss=" << result.finalLoss
              << " valLoss=" << result.valLoss
              << " threshold=" << result.anomalyThreshold;

    // Loss should decrease
    assert(result.finalLoss < result.lossHistory[0]);
    assert(result.anomalyThreshold > 0);

    std::cout << " ✓\n";
}

// ===== Ternary Quantization Training =====

void testTernaryTraining() {
    ml::Network net;
    net.setSeed(42);
    net.buildAutoencoder(4, {16, 4});

    std::vector<ml::TelemetrySample> data;
    for (int i = 0; i < 500; ++i) {
        ml::TelemetrySample sample;
        sample.timestamp = i;
        float t = i * 0.1f;
        sample.values = {
            std::sin(t) * 10.0f + 50.0f,
            std::cos(t) * 5.0f + 25.0f,
            20.0f + std::sin(t * 0.5f) * 2.0f,
            100.0f + std::cos(t * 0.3f) * 8.0f
        };
        data.push_back(sample);
    }

    ml::TrainConfig cfg;
    cfg.quant = ml::QuantMode::TERNARY;  // QAT mode
    cfg.lr = 0.001f;
    cfg.epochs = 50;
    cfg.batchSize = 32;
    cfg.patience = 20;

    ml::Trainer trainer;
    auto result = trainer.trainAutoencoder(net, data, cfg);

    // Check that ternary quantization produces valid weights
    for (const auto& layer : net.layers()) {
        net.layers()[0];  // just access
        float meanAbs = layer.weights.meanAbs();
        assert(meanAbs > 0);  // weights should be non-trivial
    }

    std::cout << "  Ternary QAT: epochs=" << result.epochsTrained
              << " loss=" << result.finalLoss
              << " ✓\n";
}

// ===== XTCE Parser =====

void testXTCEParser() {
    std::string xtce =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<SpaceSystem name=\"TestSatellite\" xmlns=\"http://www.omg.org/space/xtce\">"
        "  <TelemetryMetaData>"
        "    <ParameterTypeSet>"
        "      <FloatParameterType name=\"VoltageType\" units=\"V\">"
        "        <DefaultAlarm>"
        "          <StaticAlarmRanges>"
        "            <WarningRange minInclusive=\"22.0\" maxInclusive=\"32.0\"/>"
        "            <CriticalRange minInclusive=\"20.0\" maxInclusive=\"34.0\"/>"
        "          </StaticAlarmRanges>"
        "        </DefaultAlarm>"
        "      </FloatParameterType>"
        "      <FloatParameterType name=\"TemperatureType\" units=\"degC\">"
        "        <DefaultAlarm>"
        "          <StaticAlarmRanges>"
        "            <WarningRange minInclusive=\"-10.0\" maxInclusive=\"50.0\"/>"
        "            <CriticalRange minInclusive=\"-20.0\" maxInclusive=\"60.0\"/>"
        "          </StaticAlarmRanges>"
        "        </DefaultAlarm>"
        "        <PolynomialCalibrator>"
        "          <Term coefficient=\"-40.0\" exponent=\"0\"/>"
        "          <Term coefficient=\"0.1\" exponent=\"1\"/>"
        "        </PolynomialCalibrator>"
        "      </FloatParameterType>"
        "      <FloatParameterType name=\"CurrentType\" units=\"A\">"
        "        <DefaultAlarm>"
        "          <StaticAlarmRanges>"
        "            <WarningRange minInclusive=\"0.0\" maxInclusive=\"5.0\"/>"
        "            <CriticalRange minInclusive=\"-0.5\" maxInclusive=\"6.0\"/>"
        "          </StaticAlarmRanges>"
        "        </DefaultAlarm>"
        "      </FloatParameterType>"
        "    </ParameterTypeSet>"
        "    <ParameterSet>"
        "      <Parameter name=\"BUS_VOLTAGE\" parameterTypeRef=\"VoltageType\"/>"
        "      <Parameter name=\"PANEL_TEMP\" parameterTypeRef=\"TemperatureType\"/>"
        "      <Parameter name=\"BUS_CURRENT\" parameterTypeRef=\"CurrentType\"/>"
        "    </ParameterSet>"
        "    <ContainerSet>"
        "      <SequenceContainer name=\"PowerTM\">"
        "        <EntryList>"
        "          <ParameterRefEntry parameterRef=\"BUS_VOLTAGE\"/>"
        "          <ParameterRefEntry parameterRef=\"PANEL_TEMP\"/>"
        "          <ParameterRefEntry parameterRef=\"BUS_CURRENT\"/>"
        "        </EntryList>"
        "      </SequenceContainer>"
        "    </ContainerSet>"
        "  </TelemetryMetaData>"
        "</SpaceSystem>";

    auto def = ml::parseXTCE(xtce);

    assert(def.name == "TestSatellite");
    assert(def.parameters.size() == 3);
    assert(def.containers.size() == 1);

    // Check parameters
    assert(def.parameters[0].name == "BUS_VOLTAGE");
    assert(def.parameters[0].hasLimits);
    assertNear(def.parameters[0].warnLow, 22.0f, 0.1f, "warnLow");
    assertNear(def.parameters[0].warnHigh, 32.0f, 0.1f, "warnHigh");
    assertNear(def.parameters[0].critLow, 20.0f, 0.1f, "critLow");
    assertNear(def.parameters[0].critHigh, 34.0f, 0.1f, "critHigh");

    // Check calibration
    assert(def.parameters[1].name == "PANEL_TEMP");
    assert(def.parameters[1].calibCoeffs.size() == 2);
    float calibrated = ml::calibrate(500.0f, def.parameters[1]);
    assertNear(calibrated, 10.0f, 0.1f, "calibration");  // -40 + 0.1*500 = 10

    // Check limits
    assert(ml::checkLimits(27.0f, def.parameters[0]) == 0);  // nominal
    assert(ml::checkLimits(21.0f, def.parameters[0]) == 1);  // warning
    assert(ml::checkLimits(19.0f, def.parameters[0]) == 2);  // critical

    // Check container
    assert(def.containers[0].name == "PowerTM");
    assert(def.containers[0].parameterRefs.size() == 3);

    std::cout << "  XTCE parser: " << def.parameters.size() << " params, "
              << def.containers.size() << " containers ✓\n";
}

// ===== Synthetic Data Generation =====

void testSyntheticData() {
    std::string xtce =
        "<SpaceSystem name=\"Test\">"
        "  <TelemetryMetaData>"
        "    <ParameterTypeSet>"
        "      <FloatParameterType name=\"VoltType\">"
        "        <DefaultAlarm><StaticAlarmRanges>"
        "          <WarningRange minInclusive=\"22\" maxInclusive=\"32\"/>"
        "          <CriticalRange minInclusive=\"20\" maxInclusive=\"34\"/>"
        "        </StaticAlarmRanges></DefaultAlarm>"
        "      </FloatParameterType>"
        "      <FloatParameterType name=\"TempType\">"
        "        <DefaultAlarm><StaticAlarmRanges>"
        "          <WarningRange minInclusive=\"-10\" maxInclusive=\"50\"/>"
        "          <CriticalRange minInclusive=\"-20\" maxInclusive=\"60\"/>"
        "        </StaticAlarmRanges></DefaultAlarm>"
        "      </FloatParameterType>"
        "    </ParameterTypeSet>"
        "    <ParameterSet>"
        "      <Parameter name=\"V\" parameterTypeRef=\"VoltType\"/>"
        "      <Parameter name=\"T\" parameterTypeRef=\"TempType\"/>"
        "    </ParameterSet>"
        "  </TelemetryMetaData>"
        "</SpaceSystem>";

    auto def = ml::parseXTCE(xtce);
    auto data = ml::generateSyntheticTelemetry(def, 1000, 0.05f, 42);

    assert(data.size() == 1000);
    assert(data[0].values.size() == 2);

    // Check that values are in reasonable range
    float vSum = 0, tSum = 0;
    for (const auto& s : data) {
        vSum += s.values[0];
        tSum += s.values[1];
    }
    float vMean = vSum / 1000;
    float tMean = tSum / 1000;

    // Means should be near center of warning limits
    assertNear(vMean, 27.0f, 5.0f, "voltage mean");
    assertNear(tMean, 20.0f, 10.0f, "temp mean");

    std::cout << "  Synthetic data: " << data.size() << " samples"
              << " vMean=" << vMean << " tMean=" << tMean << " ✓\n";
}

// ===== End-to-End: XTCE → Train → Serialize → Deserialize =====

void testEndToEnd() {
    // 1. Parse XTCE
    std::string xtce =
        "<SpaceSystem name=\"CubeSat\">"
        "  <TelemetryMetaData>"
        "    <ParameterTypeSet>"
        "      <FloatParameterType name=\"VoltType\">"
        "        <DefaultAlarm><StaticAlarmRanges>"
        "          <WarningRange minInclusive=\"22\" maxInclusive=\"32\"/>"
        "          <CriticalRange minInclusive=\"20\" maxInclusive=\"34\"/>"
        "        </StaticAlarmRanges></DefaultAlarm>"
        "      </FloatParameterType>"
        "      <FloatParameterType name=\"TempType\">"
        "        <DefaultAlarm><StaticAlarmRanges>"
        "          <WarningRange minInclusive=\"-10\" maxInclusive=\"50\"/>"
        "          <CriticalRange minInclusive=\"-20\" maxInclusive=\"60\"/>"
        "        </StaticAlarmRanges></DefaultAlarm>"
        "      </FloatParameterType>"
        "      <FloatParameterType name=\"CurrType\">"
        "        <DefaultAlarm><StaticAlarmRanges>"
        "          <WarningRange minInclusive=\"0\" maxInclusive=\"5\"/>"
        "          <CriticalRange minInclusive=\"-0.5\" maxInclusive=\"6\"/>"
        "        </StaticAlarmRanges></DefaultAlarm>"
        "      </FloatParameterType>"
        "    </ParameterTypeSet>"
        "    <ParameterSet>"
        "      <Parameter name=\"V\" parameterTypeRef=\"VoltType\"/>"
        "      <Parameter name=\"T\" parameterTypeRef=\"TempType\"/>"
        "      <Parameter name=\"I\" parameterTypeRef=\"CurrType\"/>"
        "    </ParameterSet>"
        "  </TelemetryMetaData>"
        "</SpaceSystem>";

    auto def = ml::parseXTCE(xtce);
    assert(def.parameters.size() == 3);

    // 2. Generate training data
    auto data = ml::generateSyntheticTelemetry(def, 1000, 0.0f, 42);  // all normal

    // 3. Build and train autoencoder
    ml::Network net;
    net.setSeed(42);
    net.buildAutoencoder(3, {16, 4});

    ml::TrainConfig cfg;
    cfg.arch = ml::ArchType::AUTOENCODER;
    cfg.quant = ml::QuantMode::TERNARY;
    cfg.lr = 0.005f;
    cfg.epochs = 100;
    cfg.batchSize = 32;
    cfg.patience = 30;

    ml::Trainer trainer;
    auto result = trainer.trainAutoencoder(net, data, cfg);

    std::cout << "  E2E train: loss=" << result.finalLoss
              << " threshold=" << result.anomalyThreshold
              << " epochs=" << result.epochsTrained << "\n";

    // 4. Serialize to binary
    auto normParams = std::vector<ml::NormParams>();
    for (const auto& np : trainer.normalizer().params)
        normParams.push_back(np);

    auto binary = ml::serializeModel(
        net, ml::ArchType::AUTOENCODER, ml::QuantMode::TERNARY,
        normParams, result.finalLoss, result.epochsTrained,
        result.anomalyThreshold);

    std::cout << "  E2E model size: " << binary.size() << " bytes";

    // Ternary should be much smaller than float32
    auto binaryFP = ml::serializeModel(
        net, ml::ArchType::AUTOENCODER, ml::QuantMode::FULL_PRECISION,
        normParams, result.finalLoss, result.epochsTrained,
        result.anomalyThreshold);

    float ratio = (float)binaryFP.size() / binary.size();
    std::cout << " (fp32=" << binaryFP.size() << " bytes, ratio=" << ratio << "x)";

    assert(binary.size() < binaryFP.size());

    // 5. Deserialize
    ml::Network loaded;
    ml::ModelHeader header;
    std::vector<ml::NormParams> loadedNorm;
    float loadedThreshold;

    bool ok = ml::deserializeModel(binary, loaded, header, loadedNorm, loadedThreshold);
    assert(ok);
    assert(header.magic == ml::MODEL_MAGIC);
    assert(header.numLayers == net.numLayers());
    assert(header.inputDim == 3);
    assert(header.archType == (uint8_t)ml::ArchType::AUTOENCODER);
    assert(header.quantMode == (uint8_t)ml::QuantMode::TERNARY);
    assertNear(loadedThreshold, result.anomalyThreshold, 1e-5f, "threshold");

    // 6. Run inference on loaded model — normal sample should have low error
    ml::Tensor normalInput(1, 3);
    normalInput[0] = (27.0f - normParams[0].mean) / normParams[0].stddev;
    normalInput[1] = (20.0f - normParams[1].mean) / normParams[1].stddev;
    normalInput[2] = (2.5f - normParams[2].mean) / normParams[2].stddev;

    auto normalOutput = loaded.forward(normalInput);
    float normalError = normalInput.mse(normalOutput);

    // Anomalous sample (way out of range)
    ml::Tensor anomInput(1, 3);
    anomInput[0] = (50.0f - normParams[0].mean) / normParams[0].stddev;  // way high voltage
    anomInput[1] = (80.0f - normParams[1].mean) / normParams[1].stddev;  // extreme temp
    anomInput[2] = (10.0f - normParams[2].mean) / normParams[2].stddev;  // high current

    auto anomOutput = loaded.forward(anomInput);
    float anomError = anomInput.mse(anomOutput);

    std::cout << "\n  E2E inference: normal_err=" << normalError
              << " anom_err=" << anomError
              << " threshold=" << loadedThreshold;

    // Anomalous should have higher error than normal
    assert(anomError > normalError);

    std::cout << " ✓\n";
}

// ===== Model Size Estimation =====

void testModelSize() {
    // Typical CubeSat config: 10 params, [32, 8] bottleneck
    // 10→32→8→32→10
    std::vector<ml::LayerDesc> layers = {
        {10, 32, ml::Activation::RELU, true},
        {32, 8, ml::Activation::RELU, true},
        {8, 32, ml::Activation::RELU, true},
        {32, 10, ml::Activation::NONE, true}
    };

    uint32_t ternarySize = ml::estimateModelSize(layers, ml::QuantMode::TERNARY);
    uint32_t fp32Size = ml::estimateModelSize(layers, ml::QuantMode::FULL_PRECISION);

    std::cout << "  Model size (10→32→8→32→10): ternary=" << ternarySize
              << " fp32=" << fp32Size << " ratio=" << (float)fp32Size / ternarySize << "x ✓\n";

    // Larger: 50 params, [128, 32]
    layers = {
        {50, 128, ml::Activation::RELU, true},
        {128, 32, ml::Activation::RELU, true},
        {32, 128, ml::Activation::RELU, true},
        {128, 50, ml::Activation::NONE, true}
    };

    ternarySize = ml::estimateModelSize(layers, ml::QuantMode::TERNARY);
    fp32Size = ml::estimateModelSize(layers, ml::QuantMode::FULL_PRECISION);

    std::cout << "  Model size (50→128→32→128→50): ternary=" << ternarySize
              << " fp32=" << fp32Size << " ratio=" << (float)fp32Size / ternarySize << "x ✓\n";
}

}  // namespace

int main() {
    std::cout << "=== test_ml_train ===\n";
    testTensorBasics();
    testMatmul();
    testTranspose();
    testTernaryPacking();
    testForwardPass();
    testAutoencoderTraining();
    testTernaryTraining();
    testXTCEParser();
    testSyntheticData();
    testEndToEnd();
    testModelSize();
    std::cout << "All ML training tests passed.\n";
    return 0;
}
