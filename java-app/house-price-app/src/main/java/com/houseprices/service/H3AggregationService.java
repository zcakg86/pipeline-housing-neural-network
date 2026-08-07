package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import com.uber.h3core.H3Core;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Aggregates sales records by H3 L9 hexagon for the map layer.
 * Returns GeoJSON-ready feature data.
 */
@ApplicationScoped
public class H3AggregationService {

    private static final Logger LOG = Logger.getLogger(H3AggregationService.class);

    @Inject PropertyStore store;

    private final H3Core h3;
    private final Map<String, List<double[]>> boundaryCache = new ConcurrentHashMap<>();

    public H3AggregationService() {
        try { h3 = H3Core.newInstance(); }
        catch (IOException e) { throw new RuntimeException("H3Core init failed", e); }
    }

    public record HexStats(
        String h3Index,
        double avgSalePrice,
        double avgNeuralPredictedPrice,
        double avgLightgbmPredictedPrice,
        double avgGnnPredictedPrice,
        double avgNeuralPctError,
        double avgLightgbmPctError,
        double avgGnnPctError,
        double avgSqft,
        double avgSqftLot,
        int    numSales,
        double avgPredStd,          // avg prediction std dev in $
        double avgPredCvPct,        // avg 95% CI width as % of predicted price
        double attnCommunity,
        double attnProperty,
        double attnTime,
        double attnMarket,
        List<double[]> boundary
    ) {}

    /**
     * Aggregate the current immutable sales snapshot by H3 cell. The caller
     * supplies a prevalidated filter so date and viewport parsing never occurs
     * in this hot per-record loop.
     */
    public List<HexStats> aggregateSales(
            String variable,
            String model,
            PropertyRequestFilter filter) {
        Map<String, Accumulator> byHex = new HashMap<>();
        int matched = 0;
        for (PropertyRecord record : store.getSalesRecords()) {
            if (!record.hasSalePrice()) continue;
            double error = errorForModel(record, model);
            if (!filter.includes(record, error)) continue;
            if (record.h3Index() == null || record.h3Index().isBlank()) continue;
            byHex.computeIfAbsent(record.h3Index(), ignored -> new Accumulator()).add(record);
            matched++;
        }

        List<HexStats> result = new ArrayList<>(byHex.size());
        for (Map.Entry<String, Accumulator> entry : byHex.entrySet()) {
            try {
                result.add(entry.getValue().finish(entry.getKey(), boundaryFor(entry.getKey())));
            } catch (Exception exception) {
                LOG.warnf("Could not get boundary for hex %s: %s",
                    entry.getKey(), exception.getMessage());
            }
        }
        LOG.debugf("Aggregated %d hexes from %d matching sales records", result.size(), matched);
        return result;
    }

    private static final class Accumulator {
        int count;
        int attentionCount;
        double salePrice;
        double neuralPrediction;
        double treePrediction;
        double gnnPrediction;
        double neuralError;
        double treeError;
        double gnnError;
        double sqft;
        double sqftLot;
        double predictionStd;
        double predictionCv;
        final double[] attention = new double[4];

        void add(PropertyRecord record) {
            count++;
            salePrice += record.salePrice();
            neuralPrediction += record.predictedPrice();
            treePrediction += record.lightgbmPredictedPrice();
            gnnPrediction += record.gnnPredictedPrice();
            neuralError += record.pctError();
            treeError += record.lightgbmPctError();
            gnnError += record.gnnPctError();
            sqft += record.sqft();
            sqftLot += record.sqftLot();
            predictionStd += record.predictionStdPrice();
            predictionCv += record.predictionCvPct();
            float[] weights = record.clsAttention();
            if (weights != null && weights.length == 4) {
                attentionCount++;
                for (int index = 0; index < attention.length; index++) {
                    attention[index] += weights[index];
                }
            }
        }

        HexStats finish(String h3Index, List<double[]> boundary) {
            double divisor = Math.max(1, count);
            double attentionDivisor = Math.max(1, attentionCount);
            return new HexStats(
                h3Index, salePrice / divisor, neuralPrediction / divisor,
                treePrediction / divisor, gnnPrediction / divisor, neuralError / divisor, treeError / divisor, gnnError / divisor,
                sqft / divisor, sqftLot / divisor, count,
                predictionStd / divisor, predictionCv / divisor,
                attention[0] / attentionDivisor, attention[1] / attentionDivisor,
                attention[2] / attentionDivisor, attention[3] / attentionDivisor,
                boundary
            );
        }
    }

    private double errorForModel(PropertyRecord record, String model) {
        return "lightgbm".equalsIgnoreCase(model) ? record.lightgbmPctError()
            : "gnn".equalsIgnoreCase(model) ? record.gnnPctError() : record.pctError();
    }

    /** Immutable cached GeoJSON boundary for a fixed H3 cell. */
    public List<double[]> boundaryFor(String h3Index) {
        return boundaryCache.computeIfAbsent(h3Index, key -> {
            List<double[]> coordinates = h3.cellToBoundary(h3.stringToH3(key)).stream()
                .map(value -> new double[]{value.lng, value.lat})
                .collect(Collectors.toCollection(ArrayList::new));
            if (!coordinates.isEmpty()) {
                double[] first = coordinates.get(0);
                coordinates.add(new double[]{first[0], first[1]});
            }
            return Collections.unmodifiableList(coordinates);
        });
    }

    /** Build a GeoJSON FeatureCollection from aggregated hex stats */
    public Map<String, Object> toGeoJson(
        List<HexStats> hexStats, String variable, String model
    ) {
        List<Map<String, Object>> features = new ArrayList<>();

        for (HexStats hex : hexStats) {
            boolean useLightgbm = "lightgbm".equalsIgnoreCase(model);
            boolean useGnn = "gnn".equalsIgnoreCase(model);
            double selectedPrediction = useLightgbm ? hex.avgLightgbmPredictedPrice()
                : useGnn ? hex.avgGnnPredictedPrice() : hex.avgNeuralPredictedPrice();
            double selectedError = useLightgbm ? hex.avgLightgbmPctError()
                : useGnn ? hex.avgGnnPctError() : hex.avgNeuralPctError();
            double displayValue = switch (variable) {
                case "sale_price"       -> hex.avgSalePrice();
                case "predicted_price"  -> selectedPrediction;
                case "predicted_price_neural" -> hex.avgNeuralPredictedPrice();
                case "predicted_price_lightgbm"
                                          -> hex.avgLightgbmPredictedPrice();
                case "sqft"             -> hex.avgSqft();
                case "sqft_lot"         -> hex.avgSqftLot();
                case "num_sales"        -> hex.numSales();
                case "pred_std"         -> hex.avgPredStd();
                case "pred_cv_pct"      -> hex.avgPredCvPct();
                case "attn_community"   -> hex.attnCommunity();
                case "attn_property"    -> hex.attnProperty();
                case "attn_time"        -> hex.attnTime();
                case "attn_market"      -> hex.attnMarket();
                default                 -> selectedError;  // pct_error
            };

            Map<String, Object> geometry = Map.of(
                "type", "Polygon",
                "coordinates", List.of(hex.boundary())
            );

            Map<String, Object> props = new LinkedHashMap<>();
            props.put("h3Index",           hex.h3Index());
            props.put("displayValue",      displayValue);
            props.put("avgSalePrice",      hex.avgSalePrice());
            props.put("avgNeuralPredictedPrice", hex.avgNeuralPredictedPrice());
            props.put("avgLightgbmPredictedPrice", hex.avgLightgbmPredictedPrice());
            props.put("avgGnnPredictedPrice", hex.avgGnnPredictedPrice());
            props.put("avgNeuralPctError", hex.avgNeuralPctError());
            props.put("avgLightgbmPctError", hex.avgLightgbmPctError());
            props.put("avgGnnPctError", hex.avgGnnPctError());
            props.put("colorModel", useGnn ? "gnn" : useLightgbm ? "lightgbm" : "neural");
            props.put("avgSqft",           hex.avgSqft());
            props.put("avgSqftLot",        hex.avgSqftLot());
            props.put("numSales",          hex.numSales());
            props.put("avgPredStd",        hex.avgPredStd());
            props.put("avgPredCvPct",      hex.avgPredCvPct());
            props.put("attnCommunity",     hex.attnCommunity());
            props.put("attnProperty",      hex.attnProperty());
            props.put("attnTime",          hex.attnTime());
            props.put("attnMarket",        hex.attnMarket());

            features.add(Map.of("type", "Feature", "geometry", geometry, "properties", props));
        }

        return Map.of("type", "FeatureCollection", "features", features);
    }
}
