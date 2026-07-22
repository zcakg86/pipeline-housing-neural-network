package com.houseprices.model;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LightGBMContributionTest {

    @Test
    void exactContributionsReconstructNativePrediction() throws Exception {
        String model;
        try (InputStream input = getClass().getClassLoader()
                .getResourceAsStream("model-artifacts/lightgbm_model.txt")) {
            assertTrue(input != null, "LightGBM text model must be packaged");
            model = new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }

        try (LGBMBooster booster = LGBMBooster.loadModelFromString(model)) {
            int featureCount = booster.getFeatureNames().length;
            float[] features = new float[featureCount];
            double prediction = booster.predictForMat(
                features, 1, featureCount, true,
                PredictionType.C_API_PREDICT_RAW_SCORE
            )[0];
            double[] contributions = booster.predictForMat(
                features, 1, featureCount, true,
                PredictionType.C_API_PREDICT_CONTRIB
            );

            assertEquals(featureCount + 1, contributions.length);
            assertEquals(prediction, Arrays.stream(contributions).sum(), 1e-8);
        }
    }
}
