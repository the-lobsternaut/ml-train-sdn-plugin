#ifndef ML_TRAIN_XTCE_PARSER_H
#define ML_TRAIN_XTCE_PARSER_H

#include "types.h"
#include <string>

namespace ml {

// ---------------------------------------------------------------------------
// Minimal XTCE Parser
//
// XTCE (XML Telemetric and Command Exchange) defines telemetry/command
// database schemas. This parser handles the core elements needed for
// ML training: parameter types, calibration, alarm ranges.
//
// NOT a full XTCE implementation — just enough to extract:
// 1. Parameter names and types
// 2. Alarm limits (for labeling normal/anomalous)
// 3. Calibration curves (for raw→engineering conversion)
// 4. Container structure (parameter grouping)
//
// Uses a minimal hand-rolled XML parser (no libxml2 dependency).
// ---------------------------------------------------------------------------

/// Parse an XTCE XML string into a telemetry definition
XTCETelemetryDef parseXTCE(const std::string& xml);

/// Parse an XTCE file
XTCETelemetryDef parseXTCEFile(const std::string& filepath);

/// Apply calibration to a raw value
float calibrate(float raw, const XTCEParameter& param);

/// Check if a value is within alarm limits
/// Returns: 0=nominal, 1=warning, 2=critical
int checkLimits(float value, const XTCEParameter& param);

/// Generate synthetic telemetry data for testing
/// Uses parameter types and limits to create realistic ranges
std::vector<TelemetrySample> generateSyntheticTelemetry(
    const XTCETelemetryDef& def,
    uint32_t numSamples,
    float anomalyRate = 0.05f,   // fraction of samples that are anomalous
    uint32_t seed = 42);

}  // namespace ml

#endif  // ML_TRAIN_XTCE_PARSER_H
