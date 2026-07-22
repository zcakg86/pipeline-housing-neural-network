package com.houseprices.model;

import ai.onnxruntime.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.time.temporal.IsoFields;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * ONNX-backed embedding model for house price prediction.
 * The ONNX model exposes three outputs:
 *   0: log_price_scaled  [batch, 1]
 *   1: log_var_scaled    [batch, 1]   — uncertainty head (log variance in scaled log-price space)
 *   2: cls_attention     [batch, 6]   — CLS token attention over [community, year, week, property, time, market]
 */
@ApplicationScoped
public class EmbeddingModel {

    private static final Logger LOG = Logger.getLogger(EmbeddingModel.class);
    private static final int BATCH_SIZE = 512;

    /** Token names matching the 6 CLS attention output positions */
    public static final String[] ATTENTION_TOKENS =
        {"community", "year", "week", "property", "time", "market"};

    /**
     * Full prediction result carrying price, uncertainty, and attention weights.
     *
     * @param predictedPrice     Price in dollars
     * @param predictionStdPrice Standard deviation in dollars
     *                           (delta method: std_price ≈ price × std_log_price)
     * @param predictionCvPct    95% CI width as % of predicted price
     *                           (3.92 × std_log_price × 100).
     *                           Price-normalised: 15% means the same regardless of
     *                           whether the property is $300k or $1.5m.
     *                           Directly interpretable as model confidence.
     * @param clsAttention       6-element attention weights over input tokens
     *                           [community, year, week, property, time, market]
     */
    public record PredictionResult(
        double  predictedPrice,
        double  predictionStdPrice,
        double  predictionCvPct,
        float[] clsAttention
    ) {
        /** Convenience: predictedPrice only, for callers that don't need extras */
        public static PredictionResult priceOnly(double price) {
            return new PredictionResult(price, 0.0, 0.0, new float[6]);
        }
    }

    /** One exact Shapley effect for a logical neural-model input group. */
    public record ShapleyGroupEffect(
        String group,
        double logContribution,
        double priceEffectPct
    ) {}

    /** One sampled feature-level effect, constrained to its exact group total. */
    public record ShapleyFeatureEffect(
        String feature,
        String group,
        double value,
        double logContribution,
        double priceEffectPct,
        double samplingStdErrorLog,
        double samplingStdErrorPct
    ) {}

    /**
     * Exact seven-player Shapley decomposition in unscaled log-price space.
     * Continuous inputs use their training means as the reference (scaled zero),
     * while categorical inputs use their explicit unknown-token embeddings.
     */
    public record ShapleyExplanation(
        double referenceLogPrice,
        double referencePrice,
        double predictedLogPrice,
        double predictedPrice,
        int evaluatedCoalitions,
        int sampledFeatureCoalitions,
        String reference,
        List<ShapleyGroupEffect> groups,
        List<ShapleyFeatureEffect> features
    ) {}

    private static final String[] SHAPLEY_GROUPS = {
        "Community", "Year", "Week", "Property", "Time", "Economics", "Local market"
    };
    private static final int FEATURE_SHAPLEY_COALITIONS = 512;

    @Inject ModelArtifacts artifacts;
    @Inject PredictionContextFactory contextFactory;
    @Inject FeatureContract featureContract;

    private OrtEnvironment env;
    private OrtSession     session;
    private boolean usesLocalMarketFeatures;
    private int propertyFeatureCount;

    @PostConstruct
    void init() {
        try {
            env = OrtEnvironment.getEnvironment();
            byte[] modelBytes;
            try (InputStream is = getClass().getClassLoader()
                    .getResourceAsStream("model-artifacts/model.onnx")) {
                if (is == null) throw new RuntimeException("model.onnx not found in resources");
                modelBytes = is.readAllBytes();
            }
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
            usesLocalMarketFeatures = session.getInputNames().contains("local_market_features");
            TensorInfo propertyInfo = (TensorInfo) session.getInputInfo()
                .get("property_features").getInfo();
            propertyFeatureCount = Math.toIntExact(propertyInfo.getShape()[1]);
            if (propertyFeatureCount != featureContract.propertyFeatures().size()
                    || !artifacts.getPropertyFeatures().equals(
                        featureContract.propertyFeatures()
                    )) {
                throw new IllegalStateException(
                    "Neural property features do not match feature_contract.json"
                );
            }
            // Attention is part of the deployment contract. PyTorch's
            // need_weights flag is baked into the ONNX graph at export time,
            // so reject stale models that were exported without it.
            long nOutputs = session.getNumOutputs();
            LOG.infof("ONNX model loaded — %d output(s)", nOutputs);
            if (nOutputs < 3 || !session.getOutputNames().contains("cls_attention")) {
                throw new IllegalStateException(
                    "ONNX model must expose cls_attention. " +
                    "Re-run export_model_for_java.py with need_weights=True."
                );
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    @PreDestroy
    void close() {
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (env != null) env.close(); }         catch (Exception ignored) {}
    }

    // ── Public predict API ────────────────────────────────────────────────────

    /** Predict with default market indicators. */
    public PredictionResult predict(String h3Index, LocalDate saleDate,
                                    double sqft, double sqftLot, double beds) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds, Double.NaN, Double.NaN,
                       FeatureContract.DEFAULT_MORTGAGE_RATE,
                       FeatureContract.DEFAULT_UNEMPLOYMENT_RATE);
    }

    /** Predict using the property coordinates required by water-aware models. */
    public PredictionResult predict(String h3Index, LocalDate saleDate,
                                    double sqft, double sqftLot, double beds,
                                    double latitude, double longitude) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds, latitude, longitude,
                       FeatureContract.DEFAULT_MORTGAGE_RATE,
                       FeatureContract.DEFAULT_UNEMPLOYMENT_RATE);
    }

    /** Predict with explicit market indicators. */
    public PredictionResult predictWithMarket(String h3Index, LocalDate saleDate,
                                    double sqft, double sqftLot, double beds,
                                    double mortgageRate, double unemploymentRate) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds, Double.NaN, Double.NaN,
                       mortgageRate, unemploymentRate);
    }

    public PredictionResult predict(String h3Index, LocalDate saleDate,
                                    double sqft, double sqftLot, double beds,
                                    double latitude, double longitude,
                                    double mortgageRate, double unemploymentRate) {
        BatchInput input = new BatchInput(
            h3Index, saleDate, sqft, sqftLot, beds, latitude, longitude,
            mortgageRate, unemploymentRate
        );
        return predictBatch(List.of(input))[0];
    }

    // ── Batch predict ─────────────────────────────────────────────────────────

    /** Batch predict — returns a PredictionResult per record. */
    public PredictionResult[] predictBatch(List<BatchInput> records) {
        return predictBatch(records, false);
    }

    /** Batch prediction with an explicit demonstration-only snapshot look-ahead option. */
    public PredictionResult[] predictBatch(List<BatchInput> records, boolean demonstrationMode) {
        return predictPrepared(contextFactory.prepareAll(records, demonstrationMode));
    }

    public PredictionResult[] predictPrepared(List<PredictionContext> contexts) {
        PredictionResult[] results = new PredictionResult[contexts.size()];
        for (int start = 0; start < contexts.size(); start += BATCH_SIZE) {
            int end = Math.min(start + BATCH_SIZE, contexts.size());
            predictBatchChunk(
                contexts.subList(start, end), results, start
            );
        }
        return results;
    }

    /**
     * Explain one prediction with exact Shapley effects over the model's seven
     * logical input groups. All 2^7 coalitions are evaluated in one ONNX batch.
     */
    public ShapleyExplanation explain(BatchInput input, boolean demonstrationMode) {
        return explainPrepared(contextFactory.prepare(input, demonstrationMode));
    }

    public ShapleyExplanation explainPrepared(PredictionContext context) {
        BatchInput input = context.input();
        final int groupCount = SHAPLEY_GROUPS.length;
        final int coalitionCount = 1 << groupCount;

        int[] actualCommunities = context.communities();
        long actualYear = context.yearIndex();
        long actualWeek = context.weekIndex();
        float[] actualProperty = context.scaledProperty();
        float actualTime = context.scaledTimeTrend();
        float[] actualMarket = context.scaledMarket();
        float[][] actualLocal = usesLocalMarketFeatures
            ? context.scaledLocalMarket() : new float[7][5];

        long[][] communities = new long[coalitionCount][7];
        long[] years = new long[coalitionCount];
        long[] weeks = new long[coalitionCount];
        float[][] properties = new float[coalitionCount][propertyFeatureCount];
        float[][] times = new float[coalitionCount][1];
        float[][] markets = new float[coalitionCount][2];
        float[][][] locals = new float[coalitionCount][7][5];

        for (int mask = 0; mask < coalitionCount; mask++) {
            boolean includeCommunity = (mask & (1 << 0)) != 0;
            boolean includeYear = (mask & (1 << 1)) != 0;
            boolean includeWeek = (mask & (1 << 2)) != 0;
            for (int ring = 0; ring < 7; ring++) {
                communities[mask][ring] = includeCommunity
                    ? actualCommunities[ring] : artifacts.getUnknownCommunityIdx();
            }
            years[mask] = includeYear ? actualYear : artifacts.getUnknownYearIdx();
            weeks[mask] = includeWeek ? actualWeek : artifacts.getUnknownWeekIdx();
            if ((mask & (1 << 3)) != 0) {
                System.arraycopy(actualProperty, 0, properties[mask], 0, propertyFeatureCount);
            }
            if ((mask & (1 << 4)) != 0) times[mask][0] = actualTime;
            if ((mask & (1 << 5)) != 0) {
                System.arraycopy(actualMarket, 0, markets[mask], 0, 2);
            }
            if (usesLocalMarketFeatures && (mask & (1 << 6)) != 0) {
                for (int ring = 0; ring < 7; ring++) {
                    System.arraycopy(actualLocal[ring], 0, locals[mask][ring], 0, 5);
                }
            }
        }

        double[] coalitionValues = runLogPriceBatch(
            communities, years, weeks, properties, times, markets, locals
        );
        List<ShapleyGroupEffect> effects = new ArrayList<>(groupCount);
        double reconstructed = coalitionValues[0];
        for (int player = 0; player < groupCount; player++) {
            double effect = 0.0;
            int playerBit = 1 << player;
            for (int mask = 0; mask < coalitionCount; mask++) {
                if ((mask & playerBit) != 0) continue;
                int coalitionSize = Integer.bitCount(mask);
                double weight = shapleyWeight(coalitionSize, groupCount);
                effect += weight * (coalitionValues[mask | playerBit] - coalitionValues[mask]);
            }
            reconstructed += effect;
            effects.add(new ShapleyGroupEffect(
                SHAPLEY_GROUPS[player], effect, Math.expm1(effect) * 100.0
            ));
        }
        double predicted = coalitionValues[coalitionCount - 1];
        if (Math.abs(reconstructed - predicted) > 1e-5) {
            throw new IllegalStateException(String.format(
                "Neural Shapley effects do not reconstruct prediction: %.8f != %.8f",
                reconstructed, predicted
            ));
        }
        List<ShapleyFeatureEffect> featureEffects = sampledFeatureEffects(
            input, effects, actualCommunities, actualYear, actualWeek,
            actualProperty, context.rawProperty(), actualTime, actualMarket,
            actualLocal, context.rawLocalMarket()
        );
        return new ShapleyExplanation(
            coalitionValues[0], Math.exp(coalitionValues[0]),
            predicted, Math.exp(predicted), coalitionCount, FEATURE_SHAPLEY_COALITIONS,
            "continuous training means and categorical unknown tokens",
            List.copyOf(effects), featureEffects
        );
    }

    /**
     * Estimate feature-level KernelSHAP effects with one deterministic 512-row
     * ONNX batch. The estimates are projected onto the exact seven-group totals,
     * retaining feature detail while guaranteeing exact reconstruction.
     */
    private List<ShapleyFeatureEffect> sampledFeatureEffects(
            BatchInput input,
            List<ShapleyGroupEffect> exactGroups,
            int[] actualCommunities, long actualYear, long actualWeek,
            float[] actualProperty, double[] rawProperty, float actualTime,
            float[] actualMarket, float[][] actualLocal, float[][] rawLocal) {
        List<String> names = new ArrayList<>();
        List<Integer> groupIndices = new ArrayList<>();
        List<Double> displayValues = new ArrayList<>();

        for (int ring = 0; ring < 7; ring++) {
            names.add(ring == 0 ? "community_center" : "community_neighbor_" + ring);
            groupIndices.add(0);
            displayValues.add((double) actualCommunities[ring]);
        }
        names.add("year");
        groupIndices.add(1);
        displayValues.add((double) input.saleDate().getYear());
        names.add("week");
        groupIndices.add(2);
        displayValues.add((double) input.saleDate().get(IsoFields.WEEK_OF_WEEK_BASED_YEAR));

        for (int feature = 0; feature < propertyFeatureCount; feature++) {
            names.add(artifacts.getPropertyFeatures().get(feature));
            groupIndices.add(3);
            displayValues.add(rawProperty[feature]);
        }
        names.add("time_trend");
        groupIndices.add(4);
        displayValues.add(ChronoUnit.DAYS.between(
            artifacts.getReferenceDate(), input.saleDate()
        ) / 365.25);
        names.add("mortgage_rate");
        groupIndices.add(5);
        displayValues.add(input.mortgageRate());
        names.add("unemployment_rate");
        groupIndices.add(5);
        displayValues.add(input.unemploymentRate());

        if (usesLocalMarketFeatures) {
            for (int ring = 0; ring < 7; ring++) {
                String prefix = ring == 0 ? "center_" : "neighbor_" + ring + "_";
                for (int feature = 0;
                        feature < featureContract.localMarketFeatures().size(); feature++) {
                    names.add(prefix + featureContract.localMarketFeatures().get(feature));
                    groupIndices.add(6);
                    displayValues.add((double) rawLocal[ring][feature]);
                }
            }
        }

        int featureCount = names.size();
        boolean[][] masks = kernelShapleyMasks(featureCount, input);
        int sampleCount = masks.length;
        long[][] communities = new long[sampleCount][7];
        long[] years = new long[sampleCount];
        long[] weeks = new long[sampleCount];
        float[][] properties = new float[sampleCount][propertyFeatureCount];
        float[][] times = new float[sampleCount][1];
        float[][] markets = new float[sampleCount][2];
        float[][][] locals = new float[sampleCount][7][5];
        for (int row = 0; row < sampleCount; row++) {
            Arrays.fill(communities[row], artifacts.getUnknownCommunityIdx());
            years[row] = artifacts.getUnknownYearIdx();
            weeks[row] = artifacts.getUnknownWeekIdx();
            int position = 0;
            for (int ring = 0; ring < 7; ring++, position++) {
                if (masks[row][position]) communities[row][ring] = actualCommunities[ring];
            }
            if (masks[row][position++]) years[row] = actualYear;
            if (masks[row][position++]) weeks[row] = actualWeek;
            for (int feature = 0; feature < propertyFeatureCount; feature++, position++) {
                if (masks[row][position]) properties[row][feature] = actualProperty[feature];
            }
            if (masks[row][position++]) times[row][0] = actualTime;
            for (int feature = 0; feature < 2; feature++, position++) {
                if (masks[row][position]) markets[row][feature] = actualMarket[feature];
            }
            if (usesLocalMarketFeatures) {
                for (int ring = 0; ring < 7; ring++) {
                    for (int feature = 0; feature < 5; feature++, position++) {
                        if (masks[row][position]) {
                            locals[row][ring][feature] = actualLocal[ring][feature];
                        }
                    }
                }
            }
            if (position != featureCount) {
                throw new IllegalStateException("Neural feature Shapley input count mismatch");
            }
        }

        double[] values = runLogPriceBatch(
            communities, years, weeks, properties, times, markets, locals
        );
        RegressionEstimate estimate = fitKernelShapley(masks, values);
        double[] contributions = estimate.coefficients().clone();
        double[] standardErrors = estimate.standardErrors().clone();

        // Minimum-L2 projection of sampled feature estimates onto each exact
        // group total. This preserves within-group differences and efficiency.
        for (int group = 0; group < SHAPLEY_GROUPS.length; group++) {
            int count = 0;
            double sampledTotal = 0.0;
            for (int feature = 0; feature < featureCount; feature++) {
                if (groupIndices.get(feature) == group) {
                    count++;
                    sampledTotal += contributions[feature];
                }
            }
            if (count == 0) continue;
            double correction = (exactGroups.get(group).logContribution() - sampledTotal) / count;
            for (int feature = 0; feature < featureCount; feature++) {
                if (groupIndices.get(feature) == group) {
                    contributions[feature] += correction;
                    if (count == 1) standardErrors[feature] = 0.0;
                    else standardErrors[feature] *= Math.sqrt(1.0 - 1.0 / count);
                }
            }
        }

        List<ShapleyFeatureEffect> result = new ArrayList<>(featureCount);
        for (int feature = 0; feature < featureCount; feature++) {
            double contribution = contributions[feature];
            double standardError = standardErrors[feature];
            result.add(new ShapleyFeatureEffect(
                names.get(feature), SHAPLEY_GROUPS[groupIndices.get(feature)],
                displayValues.get(feature), contribution, Math.expm1(contribution) * 100.0,
                standardError, Math.exp(contribution) * standardError * 100.0
            ));
        }
        return List.copyOf(result);
    }

    private boolean[][] kernelShapleyMasks(int featureCount, BatchInput input) {
        boolean[][] masks = new boolean[FEATURE_SHAPLEY_COALITIONS][featureCount];
        Arrays.fill(masks[1], true);
        long seed = Objects.hash(
            input.h3Index(), input.saleDate(), input.sqft(), input.sqftLot(), input.beds(),
            input.latitude(), input.longitude(), input.mortgageRate(), input.unemploymentRate()
        );
        SplittableRandom random = new SplittableRandom(seed);
        int[] indices = new int[featureCount];
        for (int index = 0; index < featureCount; index++) indices[index] = index;
        for (int row = 2; row < FEATURE_SHAPLEY_COALITIONS; row += 2) {
            int coalitionSize = sampleKernelCoalitionSize(featureCount, random);
            for (int index = featureCount - 1; index > 0; index--) {
                int swap = random.nextInt(index + 1);
                int value = indices[index];
                indices[index] = indices[swap];
                indices[swap] = value;
            }
            for (int index = 0; index < coalitionSize; index++) {
                masks[row][indices[index]] = true;
            }
            if (row + 1 < FEATURE_SHAPLEY_COALITIONS) {
                for (int feature = 0; feature < featureCount; feature++) {
                    masks[row + 1][feature] = !masks[row][feature];
                }
            }
        }
        return masks;
    }

    private static int sampleKernelCoalitionSize(
            int featureCount, SplittableRandom random) {
        double total = 0.0;
        for (int size = 1; size < featureCount; size++) {
            total += 1.0 / (size * (double) (featureCount - size));
        }
        double draw = random.nextDouble(total);
        for (int size = 1; size < featureCount; size++) {
            draw -= 1.0 / (size * (double) (featureCount - size));
            if (draw <= 0.0) return size;
        }
        return featureCount - 1;
    }

    private record RegressionEstimate(double[] coefficients, double[] standardErrors) {}

    private static RegressionEstimate fitKernelShapley(
            boolean[][] masks, double[] values) {
        int featureCount = masks[0].length;
        double[][] normal = new double[featureCount][featureCount];
        double[] target = new double[featureCount];
        double baseline = values[0];
        // Sampling coalition sizes from the aggregate SHAP kernel makes the
        // non-endpoint rows an importance sample of the KernelSHAP objective.
        for (int row = 2; row < masks.length; row++) {
            double response = values[row] - baseline;
            for (int left = 0; left < featureCount; left++) {
                if (!masks[row][left]) continue;
                target[left] += response;
                for (int right = left; right < featureCount; right++) {
                    if (masks[row][right]) normal[left][right] += 1.0;
                }
            }
        }
        for (int left = 0; left < featureCount; left++) {
            for (int right = 0; right < left; right++) normal[left][right] = normal[right][left];
            normal[left][left] += 1e-6;
        }
        double[] coefficients = solveLinearSystem(normal, target);
        double squaredError = 0.0;
        int fittedRows = 0;
        for (int row = 2; row < masks.length; row++) {
            double fitted = 0.0;
            for (int feature = 0; feature < featureCount; feature++) {
                if (masks[row][feature]) fitted += coefficients[feature];
            }
            double residual = values[row] - baseline - fitted;
            squaredError += residual * residual;
            fittedRows++;
        }
        double variance = squaredError / Math.max(1, fittedRows - featureCount);
        double[] standardErrors = new double[featureCount];
        for (int feature = 0; feature < featureCount; feature++) {
            double[] unit = new double[featureCount];
            unit[feature] = 1.0;
            double diagonal = solveLinearSystem(normal, unit)[feature];
            standardErrors[feature] = Math.sqrt(Math.max(0.0, variance * diagonal));
        }
        return new RegressionEstimate(coefficients, standardErrors);
    }

    private static double[] solveLinearSystem(double[][] source, double[] sourceTarget) {
        int size = sourceTarget.length;
        double[][] matrix = new double[size][size];
        for (int row = 0; row < size; row++) matrix[row] = source[row].clone();
        double[] target = sourceTarget.clone();
        for (int pivot = 0; pivot < size; pivot++) {
            int best = pivot;
            for (int row = pivot + 1; row < size; row++) {
                if (Math.abs(matrix[row][pivot]) > Math.abs(matrix[best][pivot])) best = row;
            }
            if (Math.abs(matrix[best][pivot]) < 1e-12) {
                throw new IllegalStateException("KernelSHAP regression is singular");
            }
            double[] rowSwap = matrix[pivot];
            matrix[pivot] = matrix[best];
            matrix[best] = rowSwap;
            double valueSwap = target[pivot];
            target[pivot] = target[best];
            target[best] = valueSwap;
            for (int row = pivot + 1; row < size; row++) {
                double factor = matrix[row][pivot] / matrix[pivot][pivot];
                if (factor == 0.0) continue;
                target[row] -= factor * target[pivot];
                for (int column = pivot; column < size; column++) {
                    matrix[row][column] -= factor * matrix[pivot][column];
                }
            }
        }
        double[] solution = new double[size];
        for (int row = size - 1; row >= 0; row--) {
            double value = target[row];
            for (int column = row + 1; column < size; column++) {
                value -= matrix[row][column] * solution[column];
            }
            solution[row] = value / matrix[row][row];
        }
        return solution;
    }

    private double[] runLogPriceBatch(
        long[][] communities, long[] years, long[] weeks,
        float[][] properties, float[][] times, float[][] markets,
        float[][][] locals
    ) {
        try (OnnxTensor tCommunity = OnnxTensor.createTensor(env, communities);
             OnnxTensor tYear = OnnxTensor.createTensor(env, years);
             OnnxTensor tWeek = OnnxTensor.createTensor(env, weeks);
             OnnxTensor tProperty = OnnxTensor.createTensor(env, properties);
             OnnxTensor tTime = OnnxTensor.createTensor(env, times);
             OnnxTensor tMarket = OnnxTensor.createTensor(env, markets);
             OnnxTensor tLocal = OnnxTensor.createTensor(env, locals)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("community_indices", tCommunity);
            inputs.put("year", tYear);
            inputs.put("week", tWeek);
            inputs.put("property_features", tProperty);
            inputs.put("time_features", tTime);
            inputs.put("market_features", tMarket);
            if (usesLocalMarketFeatures) inputs.put("local_market_features", tLocal);
            try (OrtSession.Result output = session.run(inputs)) {
                float[][] scaled = (float[][]) output.get(0).getValue();
                double[] values = new double[scaled.length];
                for (int row = 0; row < scaled.length; row++) {
                    values[row] = artifacts.inverseScaleLogPrice(scaled[row][0]);
                }
                return values;
            }
        } catch (OrtException exception) {
            throw new RuntimeException("Neural Shapley ONNX inference failed", exception);
        }
    }

    private static double shapleyWeight(int coalitionSize, int playerCount) {
        return factorial(coalitionSize) * factorial(playerCount - coalitionSize - 1)
            / factorial(playerCount);
    }

    private static double factorial(int value) {
        double result = 1.0;
        for (int factor = 2; factor <= value; factor++) result *= factor;
        return result;
    }

    private void predictBatchChunk(List<PredictionContext> contexts,
                                   PredictionResult[] destination,
                                   int destinationOffset) {
        int size = contexts.size();
        long[][] communityArr = new long[size][7];
        long[] yearArr = new long[size];
        long[] weekArr = new long[size];
        float[][] propArr = new float[size][propertyFeatureCount];
        float[][] timeArr = new float[size][1];
        float[][] marketArr = new float[size][2];
        float[][][] localMarketArr = new float[size][7][5];

        for (int row = 0; row < size; row++) {
            PredictionContext context = contexts.get(row);
            int[] neighbors = context.communities();
            for (int i = 0; i < 7; i++) communityArr[row][i] = neighbors[i];
            yearArr[row] = context.yearIndex();
            weekArr[row] = context.weekIndex();
            System.arraycopy(
                context.scaledProperty(), 0, propArr[row], 0, propertyFeatureCount
            );
            timeArr[row][0] = context.scaledTimeTrend();
            System.arraycopy(context.scaledMarket(), 0, marketArr[row], 0, 2);
            if (usesLocalMarketFeatures) {
                localMarketArr[row] = context.scaledLocalMarket();
            }
        }

        try (OnnxTensor tCommunity = OnnxTensor.createTensor(env, communityArr);
             OnnxTensor tYear = OnnxTensor.createTensor(env, yearArr);
             OnnxTensor tWeek = OnnxTensor.createTensor(env, weekArr);
             OnnxTensor tProp = OnnxTensor.createTensor(env, propArr);
             OnnxTensor tTime = OnnxTensor.createTensor(env, timeArr);
             OnnxTensor tMarket = OnnxTensor.createTensor(env, marketArr);
             OnnxTensor tLocal = OnnxTensor.createTensor(env, localMarketArr)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("community_indices", tCommunity);
            inputs.put("year", tYear);
            inputs.put("week", tWeek);
            inputs.put("property_features", tProp);
            inputs.put("time_features", tTime);
            inputs.put("market_features", tMarket);
            if (usesLocalMarketFeatures) inputs.put("local_market_features", tLocal);

            try (OrtSession.Result result = session.run(inputs)) {
                float[][] priceOut = (float[][]) result.get(0).getValue();
                float[][] logVarOut = (float[][]) result.get(1).getValue();
                float[][] attnOut = (float[][]) result.get(2).getValue();
                double logPriceScale = artifacts.getLogPriceScale();
                for (int row = 0; row < size; row++) {
                    double logPrice = artifacts.inverseScaleLogPrice(priceOut[row][0]);
                    double predictedPrice = Math.exp(logPrice);
                    double stdLogPrice = Math.sqrt(Math.exp(logVarOut[row][0])) * logPriceScale;
                    float[] attention = new float[6];
                    System.arraycopy(
                        attnOut[row], 0, attention, 0, Math.min(6, attnOut[row].length)
                    );
                    destination[destinationOffset + row] = new PredictionResult(
                        predictedPrice,
                        predictedPrice * stdLogPrice,
                        3.92 * stdLogPrice * 100.0,
                        attention
                    );
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException("ONNX batch inference failed", e);
        }
    }

    /** Lightweight input record for batch prediction. */
    public record BatchInput(
        String h3Index, LocalDate saleDate,
        double sqft, double sqftLot, double beds,
        double latitude, double longitude,
        double mortgageRate, double unemploymentRate
    ) {
        public BatchInput(
            String h3Index, LocalDate saleDate,
            double sqft, double sqftLot, double beds,
            double latitude, double longitude
        ) {
            this(
                h3Index, saleDate, sqft, sqftLot, beds, latitude, longitude,
                FeatureContract.DEFAULT_MORTGAGE_RATE,
                FeatureContract.DEFAULT_UNEMPLOYMENT_RATE
            );
        }

        public BatchInput(
            String h3Index, LocalDate saleDate,
            double sqft, double sqftLot, double beds
        ) {
            this(
                h3Index, saleDate, sqft, sqftLot, beds, Double.NaN, Double.NaN,
                FeatureContract.DEFAULT_MORTGAGE_RATE,
                FeatureContract.DEFAULT_UNEMPLOYMENT_RATE
            );
        }
    }

}
