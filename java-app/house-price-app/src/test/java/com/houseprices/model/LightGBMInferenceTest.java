package com.houseprices.model;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LightGBMInferenceTest {

    @Test
    void exportedModelProducesARealisticPriceWithTheAnnualFeatureContract() {
        ModelArtifacts artifacts = new ModelArtifacts();
        artifacts.load();
        WaterProximityService water = new WaterProximityService();
        water.load();
        FeatureContract contract = new FeatureContract();
        contract.load();

        PredictionContextFactory factory = new PredictionContextFactory();
        factory.artifacts = artifacts;
        factory.waterProximity = water;
        factory.featureContract = contract;

        LightGBMModel model = new LightGBMModel();
        model.artifacts = artifacts;
        model.contextFactory = factory;
        model.featureContract = contract;
        model.init();
        try {
            double prediction = model.predict(
                "8828d08601fffff", LocalDate.of(2026, 7, 24),
                2_000.0, 4_000.0, 3.0,
                47.77198289575247, -121.90420311214848
            );
            assertTrue(
                prediction >= 100_000.0 && prediction <= 10_000_000.0,
                "Expected a price-scale prediction, received " + prediction
            );
        } finally {
            model.close();
        }
    }
}
