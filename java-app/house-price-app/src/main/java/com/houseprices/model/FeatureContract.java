package com.houseprices.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads the generated cross-language feature contract bundled with the model.
 * Model adapters use this class as the single source of truth for feature order,
 * tensor dimensions, neighborhood width, and deployment fallback values.
 */
@ApplicationScoped
public class FeatureContract {

    /** Latest contract version; version 3 remains valid for legacy bundles. */
    public static final int SUPPORTED_VERSION = 4;
    private static final int LEGACY_SUPPORTED_VERSION = 3;
    public static final double DEFAULT_MORTGAGE_RATE = 6.5;
    public static final double DEFAULT_UNEMPLOYMENT_RATE = 4.0;

    private int neighborCount;
    private List<String> propertyFeatures;
    private List<String> timeFeatures;
    private List<String> marketFeatures;
    private List<String> localMarketFeatures;
    private List<String> lightgbmFeatureNames;
    private double defaultMortgageRate;
    private double defaultUnemploymentRate;

    @PostConstruct
    void load() {
        try (InputStream input = getClass().getClassLoader()
                .getResourceAsStream("model-artifacts/feature_contract.json")) {
            if (input == null) {
                throw new IllegalStateException("feature_contract.json not found in resources");
            }
            JsonNode root = new ObjectMapper().readTree(input);
            int version = root.path("version").asInt(-1);
            if (version != LEGACY_SUPPORTED_VERSION && version != SUPPORTED_VERSION) {
                throw new IllegalStateException(
                    "Unsupported feature contract version: " + version
                );
            }
            neighborCount = root.path("neighbor_count").asInt(-1);
            JsonNode groups = root.path("groups");
            propertyFeatures = strings(groups.path("property"));
            timeFeatures = strings(groups.path("time"));
            marketFeatures = strings(groups.path("market"));
            localMarketFeatures = strings(groups.path("local_market"));
            lightgbmFeatureNames = strings(root.path("lightgbm_feature_names"));
            defaultMortgageRate = root.path("defaults").path("mortgage_rate")
                .asDouble(Double.NaN);
            defaultUnemploymentRate = root.path("defaults").path("unemployment_rate")
                .asDouble(Double.NaN);

            validateDimensions(root.path("neural_inputs"));
            if (Double.compare(defaultMortgageRate, DEFAULT_MORTGAGE_RATE) != 0
                    || Double.compare(
                        defaultUnemploymentRate, DEFAULT_UNEMPLOYMENT_RATE
                    ) != 0) {
                throw new IllegalStateException(
                    "Java fallback values do not match feature_contract.json"
                );
            }
        } catch (Exception exception) {
            throw new RuntimeException("Failed to load feature contract", exception);
        }
    }

    /** Validate declared tensor widths against the named feature groups. */
    private void validateDimensions(JsonNode inputs) {
        requireWidth(inputs, "community_indices", 1, neighborCount);
        requireWidth(inputs, "property_features", 1, propertyFeatures.size());
        requireWidth(inputs, "time_features", 1, timeFeatures.size());
        requireWidth(inputs, "market_features", 1, marketFeatures.size());
        requireWidth(inputs, "local_market_features", 1, neighborCount);
        requireWidth(inputs, "local_market_features", 2, localMarketFeatures.size());
        int expectedLightgbm = neighborCount + propertyFeatures.size()
            + timeFeatures.size() + marketFeatures.size()
            + neighborCount * localMarketFeatures.size();
        if (lightgbmFeatureNames.size() != expectedLightgbm) {
            throw new IllegalStateException(
                "Feature contract declares " + lightgbmFeatureNames.size()
                    + " LightGBM fields; expected " + expectedLightgbm
            );
        }
    }

    private void requireWidth(JsonNode inputs, String tensor, int index, int expected) {
        int actual = inputs.path(tensor).path(index).asInt(-1);
        if (actual != expected) {
            throw new IllegalStateException(
                tensor + " dimension " + index + " is " + actual
                    + "; expected " + expected
            );
        }
    }

    private List<String> strings(JsonNode values) {
        List<String> result = new ArrayList<>();
        values.forEach(value -> result.add(value.asText()));
        return List.copyOf(result);
    }

    public int neighborCount() { return neighborCount; }
    public List<String> propertyFeatures() { return propertyFeatures; }
    public List<String> timeFeatures() { return timeFeatures; }
    public List<String> marketFeatures() { return marketFeatures; }
    public List<String> localMarketFeatures() { return localMarketFeatures; }
    public List<String> lightgbmFeatureNames() { return lightgbmFeatureNames; }
    public double defaultMortgageRate() { return defaultMortgageRate; }
    public double defaultUnemploymentRate() { return defaultUnemploymentRate; }
}
