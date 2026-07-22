package com.houseprices.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.microsoft.ml.lightgbm.PredictionType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.metarank.lightgbm4j.LGBMBooster;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.temporal.IsoFields;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** ONNX-backed LightGBM inference using the exported Python feature contract. */
@ApplicationScoped
public class LightGBMModel {

    private static final Logger LOG = Logger.getLogger(LightGBMModel.class);
    private static final int BATCH_SIZE = 2048;
    @Inject ModelArtifacts artifacts;
    @Inject PredictionContextFactory contextFactory;
    @Inject FeatureContract featureContract;

    private OrtEnvironment env;
    private OrtSession session;
    private LGBMBooster contributionBooster;
    private double conformalLogResidual95;
    private int featureCount;
    private String[] featureNames;
    private boolean usesWaterFeatures;
    private String waterFeatureName;

    public record FeatureContribution(
        String feature,
        double value,
        double logContribution,
        double priceEffectPct
    ) {}

    public record GroupContribution(
        String group,
        double logContribution,
        double priceEffectPct
    ) {}

    public record PredictionExplanation(
        double baselineLogPrice,
        double baselinePrice,
        double predictedLogPrice,
        double predictedPrice,
        List<FeatureContribution> contributions,
        List<GroupContribution> groups
    ) {}

    @PostConstruct
    void init() {
        try {
            env = OrtEnvironment.getEnvironment();
            byte[] modelBytes;
            try (InputStream input = getClass().getClassLoader()
                    .getResourceAsStream("model-artifacts/lightgbm.onnx")) {
                if (input == null) {
                    throw new IllegalStateException("lightgbm.onnx not found in resources");
                }
                modelBytes = input.readAllBytes();
            }
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
            if (!session.getInputNames().contains("features") || session.getNumOutputs() != 1) {
                throw new IllegalStateException("Unexpected LightGBM ONNX input/output contract");
            }
            try (InputStream input = getClass().getClassLoader()
                    .getResourceAsStream("model-artifacts/lightgbm_metadata.json")) {
                if (input == null) {
                    throw new IllegalStateException("lightgbm_metadata.json not found in resources");
                }
                JsonNode metadata = new ObjectMapper().readTree(input);
                featureCount = metadata.path("feature_count").asInt(-1);
                JsonNode featureNameNodes = metadata.path("feature_names");
                featureNames = new String[featureNameNodes.size()];
                usesWaterFeatures = false;
                waterFeatureName = null;
                for (int index = 0; index < featureNameNodes.size(); index++) {
                    String name = featureNameNodes.get(index).asText();
                    featureNames[index] = name;
                    if ("distance_to_water_m".equals(name) ||
                            "water_proximity".equals(name)) {
                        usesWaterFeatures = true;
                        waterFeatureName = name;
                    }
                }
                if (!List.of(featureNames).equals(featureContract.lightgbmFeatureNames())) {
                    throw new IllegalStateException(
                        "LightGBM metadata feature order does not match feature_contract.json"
                    );
                }
                long onnxFeatureCount = ((ai.onnxruntime.TensorInfo) session
                    .getInputInfo().get("features").getInfo()).getShape()[1];
                if (featureCount <= 0 || featureCount != onnxFeatureCount) {
                    throw new IllegalStateException(
                        "LightGBM metadata/ONNX feature count mismatch: " +
                        featureCount + " != " + onnxFeatureCount
                    );
                }
                conformalLogResidual95 = metadata.path("uncertainty")
                    .path("absolute_log_residual_quantile").asDouble(Double.NaN);
                if (!Double.isFinite(conformalLogResidual95) || conformalLogResidual95 <= 0) {
                    throw new IllegalStateException(
                        "LightGBM metadata is missing its held-out 95% conformal interval"
                    );
                }
            }
            try (InputStream input = getClass().getClassLoader()
                    .getResourceAsStream("model-artifacts/lightgbm_model.txt")) {
                if (input == null) {
                    throw new IllegalStateException("lightgbm_model.txt not found in resources");
                }
                String modelText = new String(input.readAllBytes(), StandardCharsets.UTF_8);
                contributionBooster = LGBMBooster.loadModelFromString(modelText);
                String[] nativeFeatureNames = contributionBooster.getFeatureNames();
                if (!java.util.Arrays.equals(featureNames, nativeFeatureNames)) {
                    throw new IllegalStateException(
                        "LightGBM text model feature order does not match metadata"
                    );
                }
            }
            LOG.infof(
                "LightGBM ONNX and TreeSHAP models loaded — %d features, water=%s, " +
                "95%% log residual=%.4f",
                featureCount, usesWaterFeatures, conformalLogResidual95
            );
        } catch (Exception e) {
            throw new RuntimeException("Failed to load LightGBM ONNX model", e);
        }
    }

    @PreDestroy
    void close() {
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (contributionBooster != null) contributionBooster.close(); }
        catch (Exception ignored) {}
    }

    public double predict(String h3Index, LocalDate saleDate,
                          double sqft, double sqftLot, double beds) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds,
            Double.NaN, Double.NaN,
            FeatureContract.DEFAULT_MORTGAGE_RATE,
            FeatureContract.DEFAULT_UNEMPLOYMENT_RATE);
    }

    public double predict(String h3Index, LocalDate saleDate,
                          double sqft, double sqftLot, double beds,
                          double latitude, double longitude) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds, latitude, longitude,
            FeatureContract.DEFAULT_MORTGAGE_RATE,
            FeatureContract.DEFAULT_UNEMPLOYMENT_RATE);
    }

    public double predictWithMarket(String h3Index, LocalDate saleDate,
                          double sqft, double sqftLot, double beds,
                          double mortgageRate, double unemploymentRate) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds,
            Double.NaN, Double.NaN, mortgageRate, unemploymentRate);
    }

    public double predict(String h3Index, LocalDate saleDate,
                          double sqft, double sqftLot, double beds,
                          double latitude, double longitude,
                          double mortgageRate, double unemploymentRate) {
        EmbeddingModel.BatchInput record = new EmbeddingModel.BatchInput(
            h3Index, saleDate, sqft, sqftLot, beds, latitude, longitude,
            mortgageRate, unemploymentRate
        );
        float[][] features = buildFeatureMatrix(
            contextFactory.prepareAll(List.of(record), false)
        );
        return Math.exp(predictLogPrices(features)[0]);
    }

    public double[] predictBatch(List<EmbeddingModel.BatchInput> records) {
        return predictBatch(records, false);
    }

    /** Batch prediction with an explicit demonstration-only snapshot look-ahead option. */
    public double[] predictBatch(List<EmbeddingModel.BatchInput> records,
                                 boolean demonstrationMode) {
        return predictPrepared(contextFactory.prepareAll(records, demonstrationMode));
    }

    public double[] predictPrepared(List<PredictionContext> contexts) {
        double[] predictions = new double[contexts.size()];
        for (int start = 0; start < contexts.size(); start += BATCH_SIZE) {
            int end = Math.min(start + BATCH_SIZE, contexts.size());
            predictBatchChunk(
                contexts.subList(start, end), predictions, start
            );
        }
        return predictions;
    }

    private void predictBatchChunk(List<PredictionContext> contexts,
                                   double[] destination,
                                   int destinationOffset) {
        float[][] features = buildFeatureMatrix(contexts);
        double[] predictedLogPrice = predictLogPrices(features);
        for (int row = 0; row < contexts.size(); row++) {
            destination[destinationOffset + row] = Math.exp(predictedLogPrice[row]);
        }
    }

    /** Exact TreeSHAP contributions for one prediction, checked against ONNX output. */
    public synchronized PredictionExplanation explain(
            EmbeddingModel.BatchInput record, boolean demonstrationMode) {
        return explainPrepared(contextFactory.prepare(record, demonstrationMode));
    }

    public synchronized PredictionExplanation explainPrepared(PredictionContext context) {
        EmbeddingModel.BatchInput record = context.input();
        float[][] features = buildFeatureMatrix(List.of(context));
        try {
            double[] rawContributions = contributionBooster.predictForMat(
                features[0], 1, featureCount, true,
                PredictionType.C_API_PREDICT_CONTRIB
            );
            if (rawContributions.length != featureCount + 1) {
                throw new IllegalStateException(
                    "Expected " + (featureCount + 1) +
                    " LightGBM contribution values but received " +
                    rawContributions.length
                );
            }

            double baseline = rawContributions[featureCount];
            double reconstructedLogPrice = baseline;
            List<FeatureContribution> contributions = new ArrayList<>(featureCount);
            Map<String, Double> groupTotals = new LinkedHashMap<>();
            for (int index = 0; index < featureCount; index++) {
                double contribution = rawContributions[index];
                reconstructedLogPrice += contribution;
                contributions.add(new FeatureContribution(
                    featureNames[index],
                    displayFeatureValue(featureNames[index], features[0][index], record),
                    contribution,
                    Math.expm1(contribution) * 100.0
                ));
                groupTotals.merge(
                    contributionGroup(featureNames[index]), contribution, Double::sum
                );
            }

            double onnxLogPrice = predictLogPrices(features)[0];
            if (Math.abs(reconstructedLogPrice - onnxLogPrice) > 1e-4) {
                throw new IllegalStateException(String.format(
                    "TreeSHAP contributions do not reconstruct ONNX output: %.8f != %.8f",
                    reconstructedLogPrice, onnxLogPrice
                ));
            }

            List<GroupContribution> groups = groupTotals.entrySet().stream()
                .map(entry -> new GroupContribution(
                    entry.getKey(), entry.getValue(), Math.expm1(entry.getValue()) * 100.0
                ))
                .toList();
            return new PredictionExplanation(
                baseline,
                Math.exp(baseline),
                onnxLogPrice,
                Math.exp(onnxLogPrice),
                List.copyOf(contributions),
                groups
            );
        } catch (Exception exception) {
            throw new RuntimeException("LightGBM TreeSHAP explanation failed", exception);
        }
    }

    private float[][] buildFeatureMatrix(List<PredictionContext> contexts) {
        float[][] features = new float[contexts.size()][featureCount];
        for (int row = 0; row < contexts.size(); row++) {
            PredictionContext context = contexts.get(row);
            EmbeddingModel.BatchInput input = context.input();
            int position = 0;
            int[] communities = context.communities();
            for (int community : communities) features[row][position++] = community;
            features[row][position++] = context.yearIndex();
            features[row][position++] = context.weekIndex();
            features[row][position++] = (float) input.sqft();
            features[row][position++] = (float) input.sqftLot();
            features[row][position++] = (float) input.beds();
            if (usesWaterFeatures) {
                features[row][position++] = (float) context.rawProperty()[3];
                features[row][position++] = (float) context.rawProperty()[4];
            }
            features[row][position++] = (float) context.rawTimeTrend();
            features[row][position++] = context.rawMarket()[0];
            features[row][position++] = context.rawMarket()[1];
            float[][] local = context.rawLocalMarket();
            for (int ring = 0; ring < 7; ring++) {
                for (int feature = 0; feature < 5; feature++) {
                    features[row][position++] = local[ring][feature];
                }
            }
            if (position != featureCount) {
                throw new IllegalStateException("LightGBM feature count mismatch");
            }
        }
        return features;
    }

    private double[] predictLogPrices(float[][] features) {
        try (OnnxTensor input = OnnxTensor.createTensor(env, features);
             OrtSession.Result output = session.run(Map.of("features", input))) {
            float[][] values = (float[][]) output.get(0).getValue();
            double[] result = new double[values.length];
            for (int row = 0; row < values.length; row++) result[row] = values[row][0];
            return result;
        } catch (OrtException exception) {
            throw new RuntimeException("LightGBM ONNX inference failed", exception);
        }
    }

    private String contributionGroup(String feature) {
        if (feature.startsWith("community_")) return "Community";
        if (feature.equals("year") || feature.equals("week") ||
                feature.equals("time_trend")) return "Time";
        if (feature.equals("sqft") || feature.equals("sqft_lot") ||
                feature.equals("beds")) return "Property";
        if (feature.equals("distance_to_water_m") || feature.equals("water_proximity") ||
                feature.equals("is_waterfront")) {
            return "Waterfront";
        }
        if (feature.equals("mortgage_rate") || feature.equals("unemployment_rate")) {
            return "Economics";
        }
        if (feature.startsWith("center_local_")) return "Local market (center)";
        if (feature.startsWith("neighbor_")) return "Local market (neighbors)";
        return "Other";
    }

    private double displayFeatureValue(
            String feature, double modelValue, EmbeddingModel.BatchInput record) {
        if (feature.equals("year")) return record.saleDate().getYear();
        if (feature.equals("week")) {
            return record.saleDate().get(IsoFields.WEEK_OF_WEEK_BASED_YEAR);
        }
        return modelValue;
    }

    public double getConformalLogResidual95() {
        return conformalLogResidual95;
    }

}
