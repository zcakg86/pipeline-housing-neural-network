package com.houseprices.model;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingModelShapleyTest {

    @Test
    void exactGroupEffectsReconstructTheNeuralPrediction() {
        ModelArtifacts artifacts = new ModelArtifacts();
        artifacts.load();
        WaterProximityService water = new WaterProximityService();
        water.load();
        FeatureContract featureContract = new FeatureContract();
        featureContract.load();
        PredictionContextFactory contextFactory = new PredictionContextFactory();
        contextFactory.artifacts = artifacts;
        contextFactory.waterProximity = water;
        contextFactory.featureContract = featureContract;
        EmbeddingModel model = new EmbeddingModel();
        model.artifacts = artifacts;
        model.contextFactory = contextFactory;
        model.featureContract = featureContract;
        model.init();

        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            "8828d08601fffff", LocalDate.of(2026, 7, 20),
            2_000.0, 4_000.0, 3.0,
            47.77198289575247, -121.90420311214848,
            6.75, 4.5
        );
        EmbeddingModel.ShapleyExplanation explanation = model.explain(input, true);

        assertEquals(128, explanation.evaluatedCoalitions());
        assertEquals(512, explanation.sampledFeatureCoalitions());
        assertEquals(7, explanation.groups().size());
        assertEquals(52, explanation.features().size());
        double reconstructed = explanation.referenceLogPrice() + explanation.groups().stream()
            .mapToDouble(EmbeddingModel.ShapleyGroupEffect::logContribution)
            .sum();
        assertEquals(explanation.predictedLogPrice(), reconstructed, 1e-5);
        double featureReconstruction = explanation.referenceLogPrice() + explanation.features()
            .stream().mapToDouble(EmbeddingModel.ShapleyFeatureEffect::logContribution).sum();
        assertEquals(explanation.predictedLogPrice(), featureReconstruction, 1e-5);
        Map<String, Double> featureGroupTotals = explanation.features().stream()
            .collect(Collectors.groupingBy(
                EmbeddingModel.ShapleyFeatureEffect::group,
                Collectors.summingDouble(EmbeddingModel.ShapleyFeatureEffect::logContribution)
            ));
        for (EmbeddingModel.ShapleyGroupEffect group : explanation.groups()) {
            assertEquals(group.logContribution(), featureGroupTotals.get(group.group()), 1e-8);
        }
        assertTrue(explanation.features().stream().allMatch(feature ->
            Double.isFinite(feature.logContribution()) &&
            Double.isFinite(feature.samplingStdErrorLog()) &&
            feature.samplingStdErrorLog() >= 0.0
        ));
        assertEquals(
            model.predictBatch(List.of(input), true)[0].predictedPrice(),
            explanation.predictedPrice(),
            1.0
        );

        model.close();
    }
}
