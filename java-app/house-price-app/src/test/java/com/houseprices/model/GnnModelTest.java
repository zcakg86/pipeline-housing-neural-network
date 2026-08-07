package com.houseprices.model;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GnnModelTest {
    @Test
    void exportedMonthlyEmbeddingAndPriceHeadProduceAPrice() {
        WaterProximityService water = new WaterProximityService();
        water.load();
        MarketIndicatorService indicators = new MarketIndicatorService();
        indicators.load();
        GnnModel model = new GnnModel();
        model.water = water;
        model.indicators = indicators;
        model.load();
        try {
            double prediction = model.predict(
                "8828d08601fffff", LocalDate.of(2025, 12, 15),
                2_000.0, 4_000.0, 3.0, 47.77198289575247, -121.90420311214848
            );
            assertTrue(prediction >= 100_000.0 && prediction <= 10_000_000.0,
                "Expected price-scale GNN prediction, received " + prediction);
        } finally {
            model.close();
        }
    }

    @Test
    void exactGroupShapleyEffectsReconstructTheGnnPrice() {
        WaterProximityService water = new WaterProximityService();
        water.load();
        MarketIndicatorService indicators = new MarketIndicatorService();
        indicators.load();
        GnnModel model = new GnnModel();
        model.water = water;
        model.indicators = indicators;
        model.load();
        try {
            GnnModel.Input input = new GnnModel.Input(
                "8828d08601fffff", LocalDate.of(2025, 12, 15),
                2_000.0, 4_000.0, 3.0, 47.77198289575247, -121.90420311214848
            );
            GnnModel.ShapleyExplanation explanation = model.explain(input);
            assertEquals(16, explanation.evaluatedCoalitions());
            assertEquals(4, explanation.groups().size());
            assertEquals(10, explanation.features().size());
            assertEquals(model.predict(input.h3Index(), input.saleDate(), input.sqft(),
                input.sqftLot(), input.beds(), input.latitude(), input.longitude()),
                explanation.predictedPrice(), 0.01);
        } finally {
            model.close();
        }
    }
}
