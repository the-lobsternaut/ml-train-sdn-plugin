#include "ml_train/xtce_parser.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>

namespace ml {

// ---------------------------------------------------------------------------
// Minimal XML Tokenizer
// Just enough to parse XTCE parameter definitions.
// Not a full XML parser — handles self-closing tags, attributes, text content.
// ---------------------------------------------------------------------------

namespace xml {

struct Attribute {
    std::string name;
    std::string value;
};

struct Element {
    std::string tag;
    std::vector<Attribute> attrs;
    std::string text;
    std::vector<Element> children;

    std::string attr(const std::string& name, const std::string& def = "") const {
        for (const auto& a : attrs)
            if (a.name == name) return a.value;
        return def;
    }

    const Element* child(const std::string& tag) const {
        for (const auto& c : children)
            if (c.tag == tag) return &c;
        return nullptr;
    }

    std::vector<const Element*> childrenByTag(const std::string& tag) const {
        std::vector<const Element*> result;
        for (const auto& c : children)
            if (c.tag == tag) result.push_back(&c);
        return result;
    }
};

static void skipWS(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(s[pos])) pos++;
}

static std::string readUntil(const std::string& s, size_t& pos, char delim) {
    size_t start = pos;
    while (pos < s.size() && s[pos] != delim) pos++;
    return s.substr(start, pos - start);
}

static std::string readQuoted(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return "";
    char quote = s[pos++];
    size_t start = pos;
    while (pos < s.size() && s[pos] != quote) pos++;
    std::string result = s.substr(start, pos - start);
    if (pos < s.size()) pos++;  // skip closing quote
    return result;
}

static Element parseElement(const std::string& s, size_t& pos);

static std::vector<Attribute> parseAttributes(const std::string& s, size_t& pos) {
    std::vector<Attribute> attrs;
    while (pos < s.size()) {
        skipWS(s, pos);
        if (pos >= s.size() || s[pos] == '>' || s[pos] == '/' || s[pos] == '?') break;

        Attribute attr;
        size_t nameStart = pos;
        while (pos < s.size() && s[pos] != '=' && !std::isspace(s[pos]) && s[pos] != '>') pos++;
        attr.name = s.substr(nameStart, pos - nameStart);

        skipWS(s, pos);
        if (pos < s.size() && s[pos] == '=') {
            pos++;
            skipWS(s, pos);
            if (pos < s.size() && (s[pos] == '"' || s[pos] == '\'')) {
                attr.value = readQuoted(s, pos);
            }
        }
        if (!attr.name.empty()) attrs.push_back(attr);
    }
    return attrs;
}

static Element parseElement(const std::string& s, size_t& pos) {
    Element elem;

    // Skip to '<'
    while (pos < s.size() && s[pos] != '<') pos++;
    if (pos >= s.size()) return elem;
    pos++;  // skip '<'

    // Skip comments, processing instructions, CDATA
    if (pos < s.size() && s[pos] == '!') {
        // Comment or DOCTYPE
        auto end = s.find("-->", pos);
        if (end == std::string::npos) end = s.find(">", pos);
        if (end != std::string::npos) pos = end + (s[end-1] == '-' ? 3 : 1);
        return elem;
    }
    if (pos < s.size() && s[pos] == '?') {
        auto end = s.find("?>", pos);
        if (end != std::string::npos) pos = end + 2;
        return elem;
    }

    // Tag name
    size_t tagStart = pos;
    while (pos < s.size() && !std::isspace(s[pos]) && s[pos] != '>' && s[pos] != '/') pos++;
    elem.tag = s.substr(tagStart, pos - tagStart);

    // Strip namespace prefix
    auto colonPos = elem.tag.find(':');
    if (colonPos != std::string::npos)
        elem.tag = elem.tag.substr(colonPos + 1);

    // Attributes
    elem.attrs = parseAttributes(s, pos);

    // Self-closing?
    skipWS(s, pos);
    if (pos < s.size() && s[pos] == '/') {
        pos++;  // skip '/'
        if (pos < s.size() && s[pos] == '>') pos++;
        return elem;
    }
    if (pos < s.size() && s[pos] == '>') pos++;

    // Children and text content
    while (pos < s.size()) {
        skipWS(s, pos);
        if (pos >= s.size()) break;

        // Check for closing tag
        if (pos + 1 < s.size() && s[pos] == '<' && s[pos + 1] == '/') {
            // Skip closing tag
            auto end = s.find('>', pos);
            if (end != std::string::npos) pos = end + 1;
            break;
        }

        // Check for child element
        if (s[pos] == '<') {
            Element child = parseElement(s, pos);
            if (!child.tag.empty())
                elem.children.push_back(std::move(child));
        } else {
            // Text content
            elem.text += readUntil(s, pos, '<');
        }
    }

    return elem;
}

static Element parse(const std::string& xml) {
    size_t pos = 0;
    Element root;

    // Skip prolog
    while (pos < xml.size()) {
        skipWS(xml, pos);
        if (pos >= xml.size()) break;

        // Try to parse element
        Element elem = parseElement(xml, pos);
        if (!elem.tag.empty()) {
            root = std::move(elem);
            break;
        }
    }

    return root;
}

}  // namespace xml

// ---------------------------------------------------------------------------
// XTCE Parsing
// ---------------------------------------------------------------------------

static XTCEParamType parseParamType(const std::string& typeName) {
    if (typeName.find("Float") != std::string::npos) return XTCEParamType::FLOAT64;
    if (typeName.find("Integer") != std::string::npos) return XTCEParamType::INT32;
    if (typeName.find("Boolean") != std::string::npos) return XTCEParamType::BOOLEAN;
    if (typeName.find("Enumerated") != std::string::npos) return XTCEParamType::ENUMERATED;
    if (typeName.find("String") != std::string::npos) return XTCEParamType::STRING;
    return XTCEParamType::FLOAT64;
}

XTCETelemetryDef parseXTCE(const std::string& xmlStr) {
    XTCETelemetryDef def;

    auto root = xml::parse(xmlStr);
    def.name = root.attr("name", "Unnamed");

    // Find TelemetryMetaData
    std::function<const xml::Element*(const xml::Element&, const std::string&)> findElement;
    findElement = [&](const xml::Element& elem, const std::string& tag) -> const xml::Element* {
        if (elem.tag == tag) return &elem;
        for (const auto& child : elem.children) {
            auto result = findElement(child, tag);
            if (result) return result;
        }
        return nullptr;
    };

    // Parse ParameterTypeSet
    auto typeSet = findElement(root, "ParameterTypeSet");
    std::map<std::string, XTCEParameter> typeMap;

    if (typeSet) {
        for (const auto& child : typeSet->children) {
            XTCEParameter param;
            param.name = child.attr("name");
            param.units = child.attr("units");

            param.type = parseParamType(child.tag);

            // Look for alarm ranges
            auto alarmRanges = findElement(child, "DefaultAlarm");
            if (!alarmRanges) alarmRanges = findElement(child, "StaticAlarmRanges");
            if (alarmRanges) {
                param.hasLimits = true;
                auto warning = findElement(*alarmRanges, "WarningRange");
                auto critical = findElement(*alarmRanges, "CriticalRange");
                if (warning) {
                    std::string lo = warning->attr("minInclusive", warning->attr("minExclusive"));
                    std::string hi = warning->attr("maxInclusive", warning->attr("maxExclusive"));
                    if (!lo.empty()) param.warnLow = std::stof(lo);
                    if (!hi.empty()) param.warnHigh = std::stof(hi);
                }
                if (critical) {
                    std::string lo = critical->attr("minInclusive", critical->attr("minExclusive"));
                    std::string hi = critical->attr("maxInclusive", critical->attr("maxExclusive"));
                    if (!lo.empty()) param.critLow = std::stof(lo);
                    if (!hi.empty()) param.critHigh = std::stof(hi);
                }
            }

            // Calibration polynomial
            auto poly = findElement(child, "PolynomialCalibrator");
            if (poly) {
                for (const auto& term : poly->children) {
                    if (term.tag == "Term") {
                        float coeff = std::stof(term.attr("coefficient", "0"));
                        int exp = std::stoi(term.attr("exponent", "0"));
                        if (param.calibCoeffs.size() <= (size_t)exp)
                            param.calibCoeffs.resize(exp + 1, 0.0f);
                        param.calibCoeffs[exp] = coeff;
                    }
                }
            }

            if (!param.name.empty())
                typeMap[param.name] = param;
        }
    }

    // Parse ParameterSet
    auto paramSet = findElement(root, "ParameterSet");
    if (paramSet) {
        for (const auto& child : paramSet->children) {
            if (child.tag != "Parameter") continue;

            std::string name = child.attr("name");
            std::string typeRef = child.attr("parameterTypeRef");

            XTCEParameter param;
            if (typeMap.count(typeRef)) {
                param = typeMap[typeRef];
            }
            param.name = name;

            // Override description
            auto desc = findElement(child, "LongDescription");
            if (desc) param.description = desc->text;

            def.parameters.push_back(param);
        }
    }

    // Parse ContainerSet
    auto containerSet = findElement(root, "ContainerSet");
    if (containerSet) {
        for (const auto& child : containerSet->children) {
            if (child.tag != "SequenceContainer") continue;

            XTCEContainer container;
            container.name = child.attr("name");

            auto entryList = findElement(child, "EntryList");
            if (entryList) {
                for (const auto& entry : entryList->children) {
                    if (entry.tag == "ParameterRefEntry") {
                        container.parameterRefs.push_back(entry.attr("parameterRef"));
                    }
                }
            }

            auto rateInStream = findElement(child, "DefaultRateInStream");
            if (rateInStream) {
                std::string minFreq = rateInStream->attr("minimumValue", "1");
                container.rateHz = std::stof(minFreq);
            }

            def.containers.push_back(container);
        }
    }

    return def;
}

XTCETelemetryDef parseXTCEFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return XTCETelemetryDef{};
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return parseXTCE(ss.str());
}

float calibrate(float raw, const XTCEParameter& param) {
    if (param.calibCoeffs.empty()) return raw;
    float result = 0;
    float x = 1.0f;
    for (auto coeff : param.calibCoeffs) {
        result += coeff * x;
        x *= raw;
    }
    return result;
}

int checkLimits(float value, const XTCEParameter& param) {
    if (!param.hasLimits) return 0;
    if (value < param.critLow || value > param.critHigh) return 2;
    if (value < param.warnLow || value > param.warnHigh) return 1;
    return 0;
}

std::vector<TelemetrySample> generateSyntheticTelemetry(
    const XTCETelemetryDef& def,
    uint32_t numSamples,
    float anomalyRate,
    uint32_t seed) {

    std::mt19937 rng(seed);
    std::vector<TelemetrySample> data;

    uint32_t dim = def.parameters.size();
    if (dim == 0) return data;

    // Use parameter limits to determine normal ranges
    std::vector<float> nominalMean(dim);
    std::vector<float> nominalStd(dim);

    for (uint32_t j = 0; j < dim; ++j) {
        const auto& p = def.parameters[j];
        if (p.hasLimits) {
            nominalMean[j] = (p.warnLow + p.warnHigh) / 2.0f;
            nominalStd[j] = (p.warnHigh - p.warnLow) / 6.0f;  // 3-sigma within warn limits
        } else {
            nominalMean[j] = 0.0f;
            nominalStd[j] = 1.0f;
        }
        if (nominalStd[j] < 1e-6f) nominalStd[j] = 1.0f;
    }

    std::uniform_real_distribution<float> anomalyDist(0.0f, 1.0f);

    for (uint32_t i = 0; i < numSamples; ++i) {
        TelemetrySample sample;
        sample.timestamp = (double)i;
        sample.values.resize(dim);
        sample.valid.resize(dim, true);

        bool isAnomaly = anomalyDist(rng) < anomalyRate;

        for (uint32_t j = 0; j < dim; ++j) {
            std::normal_distribution<float> normal(nominalMean[j], nominalStd[j]);
            float value = normal(rng);

            if (isAnomaly) {
                // Pick 1-2 parameters to make anomalous
                if (anomalyDist(rng) < 0.3f) {
                    // Push outside warning limits
                    float sign = (anomalyDist(rng) < 0.5f) ? 1.0f : -1.0f;
                    value = nominalMean[j] + sign * nominalStd[j] * (4.0f + anomalyDist(rng) * 3.0f);
                }
            }

            sample.values[j] = value;
        }

        data.push_back(sample);
    }

    return data;
}

}  // namespace ml
