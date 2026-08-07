package com.houseprices.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.DataInputStream;
import java.io.InputStream;
import java.time.LocalDate;
import java.time.YearMonth;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Java inference for the monthly H3 GraphSAGE baseline.
 *
 * Python performs GraphSAGE message passing once for each causal monthly H3
 * snapshot. Java loads those immutable cell embeddings and runs only the small
 * ONNX price head, avoiding graph reconstruction or per-request propagation.
 */
@ApplicationScoped
public class GnnModel {
    private static final Logger LOG = Logger.getLogger(GnnModel.class);
    private static final int BATCH_SIZE = 2048;
    private static final byte[] MAGIC = "HPGNN01".getBytes(java.nio.charset.StandardCharsets.US_ASCII);

    @jakarta.inject.Inject WaterProximityService water;
    @jakarta.inject.Inject MarketIndicatorService indicators;

    private OrtEnvironment environment;
    private OrtSession session;
    private Map<String, double[][]> scalers;
    private Map<String, Integer> cellIndex;
    private Map<YearMonth, Integer> monthIndex;
    private float[] embeddings;
    private int cellCount;
    private int embeddingDim;
    private int monthCount;
    private LocalDate referenceDate;
    private YearMonth latestMonth;
    private double conformalLogResidual90;
    private double conformalLogResidual95;

    /** An exact Shapley effect for one deployable GNN input group. */
    public record ShapleyGroupEffect(
        String group,
        double logContribution,
        double priceEffectPct
    ) {}

    /** One exact price-head feature effect; the spatial embedding stays atomic. */
    public record ShapleyFeatureEffect(
        String feature,
        String group,
        double value,
        double logContribution,
        double priceEffectPct
    ) {}

    /**
     * GNN explanation at the level Java can reproduce faithfully.
     *
     * The deployment artifact contains precomputed monthly GraphSAGE embeddings,
     * rather than the graph-convolution layers and their historical node states.
     * Consequently, "Spatial H3 embedding" is one indivisible feature here;
     * property, time, and economic fields can be attributed individually.
     */
    public record ShapleyExplanation(
        double referenceLogPrice,
        double referencePrice,
        double predictedLogPrice,
        double predictedPrice,
        int evaluatedCoalitions,
        String reference,
        List<ShapleyGroupEffect> groups,
        List<ShapleyFeatureEffect> features
    ) {}

    private record PreparedInput(
        float[] embedding, float[] property, float[] time, float[] market
    ) {}

    private static final String[] SHAPLEY_GROUPS = {
        "Spatial H3 embedding", "Property", "Time", "Economics"
    };
    private static final String[] FEATURE_NAMES = {
        "spatial_h3_embedding", "sqft", "sqft_lot", "beds", "water_proximity",
        "time_trend", "annual_sin", "annual_cos", "mortgage_rate", "unemployment_rate"
    };
    private static final String[] FEATURE_GROUPS = {
        "Spatial H3 embedding", "Property", "Property", "Property", "Property",
        "Time", "Time", "Time", "Economics", "Economics"
    };

    @PostConstruct
    void load() {
        try {
            ObjectMapper mapper = new ObjectMapper();
            try (InputStream input = resource("gnn_scalers.json")) {
                Map<String, Map<String, List<Double>>> raw = mapper.readValue(input,
                    new TypeReference<>() {});
                scalers = new HashMap<>();
                raw.forEach((name, value) -> scalers.put(name,
                    new double[][]{value.get("mean").stream().mapToDouble(Double::doubleValue).toArray(),
                        value.get("scale").stream().mapToDouble(Double::doubleValue).toArray()}));
            }
            try (InputStream input = resource("gnn_metadata.json")) {
                Map<String, Object> metadata = mapper.readValue(input, new TypeReference<>() {});
                referenceDate = LocalDate.parse(String.valueOf(metadata.get("reference_date")));
                Object months = metadata.get("month_starts");
                if (!(months instanceof List<?> values) || values.isEmpty()) {
                    throw new IllegalStateException("GNN metadata has no monthly snapshots");
                }
                Map<?, ?> uncertainty = metadata.get("uncertainty") instanceof Map<?, ?> value
                    ? value : Map.of();
                Map<?, ?> quantiles = uncertainty.get("absolute_log_residual_quantiles") instanceof Map<?, ?> value
                    ? value : Map.of();
                conformalLogResidual90 = metadataDouble(quantiles.get("0.90"));
                conformalLogResidual95 = metadataDouble(quantiles.get("0.95"));
                if (!Double.isFinite(conformalLogResidual90) || conformalLogResidual90 <= 0
                        || !Double.isFinite(conformalLogResidual95) || conformalLogResidual95 <= 0) {
                    throw new IllegalStateException(
                        "GNN metadata is missing held-out 90%/95% conformal intervals; re-export the GNN bundle"
                    );
                }
            }
            loadEmbeddingTable();
            environment = OrtEnvironment.getEnvironment();
            try (InputStream input = resource("gnn_price_head.onnx")) {
                session = environment.createSession(input.readAllBytes(), new OrtSession.SessionOptions());
            }
            LOG.infof("Loaded GNN price head: %,d cells × %d months × %d dimensions; latest=%s",
                cellCount, monthCount, embeddingDim, latestMonth);
        } catch (Exception exception) {
            throw new RuntimeException("Failed to load GNN model artifacts", exception);
        }
    }

    private InputStream resource(String name) {
        InputStream input = getClass().getClassLoader().getResourceAsStream("model-artifacts/" + name);
        if (input == null) throw new IllegalStateException(name + " not found in model artifacts");
        return input;
    }

    private static double metadataDouble(Object value) {
        if (value instanceof Number number) return number.doubleValue();
        try { return value == null ? Double.NaN : Double.parseDouble(String.valueOf(value)); }
        catch (NumberFormatException ignored) { return Double.NaN; }
    }

    private void loadEmbeddingTable() throws Exception {
        try (DataInputStream input = new DataInputStream(new GZIPInputStream(resource("gnn_monthly_embeddings.bin.gz")))) {
            byte[] magic = input.readNBytes(MAGIC.length);
            if (!java.util.Arrays.equals(magic, MAGIC)) throw new IllegalStateException("Invalid GNN embedding artifact");
            monthCount = input.readInt();
            cellCount = input.readInt();
            embeddingDim = input.readInt();
            monthIndex = new HashMap<>();
            for (int index = 0; index < monthCount; index++) {
                int length = input.readUnsignedShort();
                YearMonth month = YearMonth.from(LocalDate.parse(
                    new String(input.readNBytes(length), java.nio.charset.StandardCharsets.US_ASCII)
                ));
                monthIndex.put(month, index);
                latestMonth = month;
            }
            cellIndex = new HashMap<>();
            for (int index = 0; index < cellCount; index++) {
                int length = input.readUnsignedShort();
                cellIndex.put(new String(input.readNBytes(length), java.nio.charset.StandardCharsets.US_ASCII), index);
            }
            embeddings = new float[Math.multiplyExact(Math.multiplyExact(monthCount, cellCount), embeddingDim)];
            for (int index = 0; index < embeddings.length; index++) embeddings[index] = input.readFloat();
        }
    }

    /** Predict one property; future dates use the latest safe, though ageing, snapshot. */
    public double predict(String h3Index, LocalDate saleDate, double sqft, double sqftLot, double beds,
                          double latitude, double longitude) {
        return predictBatch(List.of(new Input(h3Index, saleDate, sqft, sqftLot, beds, latitude, longitude)))[0];
    }

    public double[] predictBatch(List<Input> inputs) {
        double[] result = new double[inputs.size()];
        for (int start = 0; start < inputs.size(); start += BATCH_SIZE) {
            int end = Math.min(inputs.size(), start + BATCH_SIZE);
            predictChunk(inputs.subList(start, end), result, start);
        }
        return result;
    }

    private void predictChunk(List<Input> inputs, double[] destination, int offset) {
        int rows = inputs.size();
        float[][] gnn = new float[rows][embeddingDim];
        float[][] property = new float[rows][4];
        float[][] time = new float[rows][3];
        float[][] market = new float[rows][2];
        for (int row = 0; row < rows; row++) {
            PreparedInput prepared = prepare(inputs.get(row));
            gnn[row] = prepared.embedding();
            property[row] = prepared.property();
            time[row] = prepared.time();
            market[row] = prepared.market();
        }
        double[] values = runScaledLogPriceBatch(gnn, property, time, market);
        double[][] target = scalers.get("target");
        for (int row = 0; row < rows; row++) {
            destination[offset + row] = Math.exp(values[row] * target[1][0] + target[0][0]);
        }
    }

    /**
     * Explain a deployed prediction with all 16 exact coalitions of its four
     * price-head inputs. This does not claim to decompose the frozen GraphSAGE
     * embedding into historical cell features; that would require exporting and
     * executing the full message-passing graph in Java.
     */
    public ShapleyExplanation explain(Input input) {
        PreparedInput actual = prepare(input);
        int groupCount = SHAPLEY_GROUPS.length;
        int coalitionCount = 1 << groupCount;
        float[][] embeddings = new float[coalitionCount][embeddingDim];
        float[][] properties = new float[coalitionCount][4];
        float[][] times = new float[coalitionCount][3];
        float[][] markets = new float[coalitionCount][2];
        for (int mask = 0; mask < coalitionCount; mask++) {
            if ((mask & 1) != 0) System.arraycopy(actual.embedding(), 0, embeddings[mask], 0, embeddingDim);
            if ((mask & 2) != 0) System.arraycopy(actual.property(), 0, properties[mask], 0, 4);
            if ((mask & 4) != 0) System.arraycopy(actual.time(), 0, times[mask], 0, 3);
            if ((mask & 8) != 0) System.arraycopy(actual.market(), 0, markets[mask], 0, 2);
        }
        double[] values = runScaledLogPriceBatch(embeddings, properties, times, markets);
        double[][] target = scalers.get("target");
        for (int index = 0; index < values.length; index++) {
            values[index] = values[index] * target[1][0] + target[0][0];
        }
        List<ShapleyGroupEffect> effects = new ArrayList<>(groupCount);
        double reconstructed = values[0];
        for (int player = 0; player < groupCount; player++) {
            double effect = 0.0;
            int bit = 1 << player;
            for (int mask = 0; mask < coalitionCount; mask++) {
                if ((mask & bit) == 0) {
                    effect += shapleyWeight(Integer.bitCount(mask), groupCount)
                        * (values[mask | bit] - values[mask]);
                }
            }
            reconstructed += effect;
            effects.add(new ShapleyGroupEffect(
                SHAPLEY_GROUPS[player], effect, Math.expm1(effect) * 100.0
            ));
        }
        double predicted = values[coalitionCount - 1];
        if (Math.abs(reconstructed - predicted) > 1e-5) {
            throw new IllegalStateException("GNN Shapley effects do not reconstruct prediction");
        }
        return new ShapleyExplanation(
            values[0], Math.exp(values[0]), predicted, Math.exp(predicted), coalitionCount,
            "training means for property, time, and economics; zero graph embedding for spatial context",
            List.copyOf(effects), featureEffects(input, actual)
        );
    }

    /**
     * Evaluate all 2^10 coalitions for exact, readable price-head effects.
     * Keeping the 32-dimensional GraphSAGE output together avoids assigning
     * meaning to arbitrary latent-vector coordinates.
     */
    private List<ShapleyFeatureEffect> featureEffects(Input input, PreparedInput actual) {
        int featureCount = FEATURE_NAMES.length;
        int coalitionCount = 1 << featureCount;
        float[][] embeddings = new float[coalitionCount][embeddingDim];
        float[][] properties = new float[coalitionCount][4];
        float[][] times = new float[coalitionCount][3];
        float[][] markets = new float[coalitionCount][2];
        for (int mask = 0; mask < coalitionCount; mask++) {
            if ((mask & 1) != 0) System.arraycopy(actual.embedding(), 0, embeddings[mask], 0, embeddingDim);
            for (int index = 0; index < 4; index++) {
                if ((mask & (1 << (index + 1))) != 0) properties[mask][index] = actual.property()[index];
            }
            for (int index = 0; index < 3; index++) {
                if ((mask & (1 << (index + 5))) != 0) times[mask][index] = actual.time()[index];
            }
            for (int index = 0; index < 2; index++) {
                if ((mask & (1 << (index + 8))) != 0) markets[mask][index] = actual.market()[index];
            }
        }
        double[] scaledValues = runScaledLogPriceBatch(embeddings, properties, times, markets);
        double[][] target = scalers.get("target");
        for (int index = 0; index < scaledValues.length; index++) {
            scaledValues[index] = scaledValues[index] * target[1][0] + target[0][0];
        }
        double[] raw = rawFeatureValues(input);
        List<ShapleyFeatureEffect> effects = new ArrayList<>(featureCount);
        double reconstructed = scaledValues[0];
        for (int player = 0; player < featureCount; player++) {
            int bit = 1 << player;
            double effect = 0.0;
            for (int mask = 0; mask < coalitionCount; mask++) {
                if ((mask & bit) == 0) {
                    effect += shapleyWeight(Integer.bitCount(mask), featureCount)
                        * (scaledValues[mask | bit] - scaledValues[mask]);
                }
            }
            reconstructed += effect;
            effects.add(new ShapleyFeatureEffect(
                FEATURE_NAMES[player], FEATURE_GROUPS[player], raw[player], effect,
                Math.expm1(effect) * 100.0
            ));
        }
        if (Math.abs(reconstructed - scaledValues[coalitionCount - 1]) > 1e-5) {
            throw new IllegalStateException("GNN feature Shapley effects do not reconstruct prediction");
        }
        return List.copyOf(effects);
    }

    private double[] rawFeatureValues(Input input) {
        WaterProximityService.WaterFeatures waterFeatures = water.lookup(input.latitude(), input.longitude());
        double year = java.time.Year.isLeap(input.saleDate().getYear()) ? 366.0 : 365.0;
        double phase = 2.0 * Math.PI * (input.saleDate().getDayOfYear() - 1.0) / year;
        MarketIndicatorService.MarketIndicators market = indicators.lookup(input.saleDate());
        return new double[]{0.0, input.sqft(), input.sqftLot(), input.beds(),
            WaterProximityService.waterProximity(waterFeatures.distanceToWaterM()),
            java.time.temporal.ChronoUnit.DAYS.between(referenceDate, input.saleDate()) / 365.25,
            Math.sin(phase), Math.cos(phase), market.mortgageRate(), market.unemploymentRate()};
    }

    private PreparedInput prepare(Input value) {
        float[] embedding = new float[embeddingDim];
        copyEmbedding(value.h3Index(), value.saleDate(), embedding);
        WaterProximityService.WaterFeatures waterFeatures = water.lookup(value.latitude(), value.longitude());
        float[] property = new float[]{scale("property", 0, value.sqft()), scale("property", 1, value.sqftLot()),
            scale("property", 2, value.beds()), scale("property", 3, WaterProximityService.waterProximity(waterFeatures.distanceToWaterM()))};
        double year = java.time.Year.isLeap(value.saleDate().getYear()) ? 366.0 : 365.0;
        double phase = 2.0 * Math.PI * (value.saleDate().getDayOfYear() - 1.0) / year;
        float[] time = new float[]{scale("time", 0, java.time.temporal.ChronoUnit.DAYS.between(referenceDate, value.saleDate()) / 365.25),
            scale("time", 1, Math.sin(phase)), scale("time", 2, Math.cos(phase))};
        MarketIndicatorService.MarketIndicators indicator = indicators.lookup(value.saleDate());
        float[] market = new float[]{scale("market", 0, indicator.mortgageRate()), scale("market", 1, indicator.unemploymentRate())};
        return new PreparedInput(embedding, property, time, market);
    }

    private double[] runScaledLogPriceBatch(float[][] gnn, float[][] property, float[][] time, float[][] market) {
        try (OnnxTensor gnnTensor = OnnxTensor.createTensor(environment, gnn);
             OnnxTensor propertyTensor = OnnxTensor.createTensor(environment, property);
             OnnxTensor timeTensor = OnnxTensor.createTensor(environment, time);
             OnnxTensor marketTensor = OnnxTensor.createTensor(environment, market);
             OrtSession.Result output = session.run(Map.of("gnn_embedding", gnnTensor, "property_features", propertyTensor,
                 "time_features", timeTensor, "market_features", marketTensor))) {
            float[][] values = (float[][]) output.get(0).getValue();
            double[] result = new double[values.length];
            for (int row = 0; row < values.length; row++) result[row] = values[row][0];
            return result;
        } catch (Exception exception) {
            throw new RuntimeException("GNN ONNX inference failed", exception);
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

    private float scale(String group, int field, double value) {
        String[] names = switch (group) {
            case "property" -> new String[]{"sqft", "sqft_lot", "beds", "water_proximity"};
            case "time" -> new String[]{"time_trend", "annual_sin", "annual_cos"};
            default -> new String[]{"mortgage_rate", "unemployment_rate"};
        };
        double[][] scaler = scalers.get(group);
        if (scaler == null || field >= scaler[0].length || field >= scaler[1].length) {
            throw new IllegalStateException("Missing GNN " + group + " scaler field " + field);
        }
        return (float) ((value - scaler[0][field]) / scaler[1][field]);
    }

    private void copyEmbedding(String h3Index, LocalDate date, float[] destination) {
        Integer cell = cellIndex.get(h3Index);
        if (cell == null) return; // unseen cells receive the learned zero-state fallback.
        YearMonth requested = YearMonth.from(date);
        Integer month = monthIndex.get(requested);
        if (month == null) month = requested.isAfter(latestMonth) ? monthIndex.get(latestMonth) : 0;
        int start = (month * cellCount + cell) * embeddingDim;
        System.arraycopy(embeddings, start, destination, 0, embeddingDim);
    }

    @PreDestroy void close() { try { if (session != null) session.close(); } catch (Exception ignored) {} }

    public record Input(String h3Index, LocalDate saleDate, double sqft, double sqftLot, double beds,
                        double latitude, double longitude) {}

    /** Held-out absolute-log-residual conformal radius for a nominal 90% interval. */
    public double getConformalLogResidual90() { return conformalLogResidual90; }

    /** Held-out absolute-log-residual conformal radius for a nominal 95% interval. */
    public double getConformalLogResidual95() { return conformalLogResidual95; }
}
