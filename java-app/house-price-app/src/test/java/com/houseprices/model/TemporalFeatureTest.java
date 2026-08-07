package com.houseprices.model;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TemporalFeatureTest {

    @Test
    void januaryFirstStartsTheContinuousAnnualCycle() {
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

        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            "8828d08601fffff", LocalDate.of(2027, 1, 1),
            2_000.0, 4_000.0, 3.0,
            47.77198289575247, -121.90420311214848,
            6.75, 4.5
        );
        PredictionContext context = factory.prepare(input, true);

        assertEquals(3, context.rawTime().length);
        assertEquals(0.0, context.rawTime()[1], 1e-12);
        assertEquals(1.0, context.rawTime()[2], 1e-12);
    }
}
