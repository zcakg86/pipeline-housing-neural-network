package com.houseprices.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Lagged mixed-source local context used exclusively for RentCast records.
 *
 * The generated snapshot ends three months before the newest observed sale,
 * keeping the newest sales free of future local-sale context. Historical CSV
 * records continue to use their Python-generated causal features and never
 * read this snapshot.
 */
@ApplicationScoped
public class RentcastLocalMarketService {

    private static final Logger LOG = Logger.getLogger(RentcastLocalMarketService.class);
    private final Map<String, String[]> neighborCells = new HashMap<>();
    private final Map<String, Map<String, Object>> cells = new HashMap<>();
    /** Exact rolling trends, memoized because one ingestion batch reuses cells and dates. */
    private final Map<TrendKey, OptionalDouble> cellTrendCache = new ConcurrentHashMap<>();
    private final Map<LocalDate, OptionalDouble> globalTrendCache = new ConcurrentHashMap<>();
    private Map<String, Object> global = Map.of();
    private LocalDate asOfDate;
    private LocalDate latestSaleDate;
    private double maxRecencyYears = 5.0;
    private double decayHalfLifeDays = 730.0;
    private int trendWindowDays = 365;
    private double premiumShrinkageWeight = 3.0;

    @PostConstruct
    @SuppressWarnings("unchecked")
    void load() {
        ObjectMapper mapper = new ObjectMapper();
        try (
            InputStream snapshotInput = resource("rentcast_local_market_snapshot.json");
            InputStream neighborsInput = resource("rentcast_h3_l8_neighbor_cells.json")
        ) {
            Map<String, Object> snapshot = mapper.readValue(snapshotInput, Map.class);
            Map<String, List<String>> rawNeighbors = mapper.readValue(
                neighborsInput,
                mapper.getTypeFactory().constructMapType(Map.class, String.class, List.class)
            );
            rawNeighbors.forEach((cell, ring) ->
                neighborCells.put(cell, ring.toArray(new String[0]))
            );
            Object rawCells = snapshot.get("cells");
            if (rawCells instanceof Map<?, ?> map) {
                map.forEach((key, value) -> {
                    if (value instanceof Map<?, ?> record) {
                        Map<String, Object> prepared = (Map<String, Object>) record;
                        prepareDates(prepared);
                        cells.put(String.valueOf(key), prepared);
                    }
                });
            }
            if (snapshot.get("global") instanceof Map<?, ?> record) {
                global = (Map<String, Object>) record;
                prepareDates(global);
            }
            maxRecencyYears = numeric(snapshot, "max_recency_years", maxRecencyYears);
            decayHalfLifeDays = numeric(snapshot, "decay_half_life_days", decayHalfLifeDays);
            trendWindowDays = (int) numeric(snapshot, "trend_window_days", trendWindowDays);
            premiumShrinkageWeight = numeric(
                snapshot, "premium_shrinkage_weight", premiumShrinkageWeight
            );
            asOfDate = parseDate(snapshot.get("as_of_date"));
            latestSaleDate = parseDate(snapshot.get("latest_sale_date"));
            if (cells.isEmpty() || latestSaleDate == null) {
                throw new IllegalStateException("RentCast local-market snapshot is incomplete");
            }
            LOG.infof(
                "Loaded RentCast lagged local market: %d H3 cells, latest sale=%s, as-of=%s",
                cells.size(), latestSaleDate, asOfDate
            );
        } catch (Exception exception) {
            throw new RuntimeException("Failed to load RentCast local-market snapshot", exception);
        }
    }

    /** Raw [7,5] features matching the exported model's decayed local schema. */
    public float[][] rawFeatures(String h3Index, LocalDate predictionDate) {
        if (predictionDate == null) throw new IllegalArgumentException("Prediction date is required");
        String[] ring = neighborCells.get(h3Index);
        if (ring == null || ring.length != 7) {
            ring = new String[7];
            ring[0] = h3Index;
        }
        float[][] result = new float[7][5];
        double globalStd = numeric(global, "decayed_log_price_std", 0.5);
        for (int index = 0; index < ring.length; index++) {
            Map<String, Object> cell = ring[index] == null ? null : cells.get(ring[index]);
            boolean hasCell = cell != null;
            if (cell == null) cell = global;
            double support = hasCell ? agedSupport(cell, predictionDate) : 0.0;
            double rawPremium = hasCell
                ? numeric(cell, "raw_decayed_log_price_premium", 0.0) : 0.0;
            double premium = support <= 0.0 ? 0.0
                : rawPremium * support / (support + premiumShrinkageWeight);
            double std = hasCell ? numeric(cell, "decayed_log_price_std", globalStd) : globalStd;
            double recency = recency(cell, predictionDate);
            double trend = hasCell ? relativeTrend(ring[index], cell, predictionDate) : 0.0;
            result[index][0] = (float) premium;
            result[index][1] = (float) std;
            result[index][2] = (float) Math.log1p(support);
            result[index][3] = (float) recency;
            result[index][4] = (float) trend;
        }
        return result;
    }

    /** A displayed RentCast prediction is non-causal when its sale predates this lagged state. */
    public boolean hasLookAheadIssue(LocalDate saleDate) {
        return saleDate != null && latestSaleDate != null && !saleDate.isAfter(latestSaleDate);
    }

    public LocalDate latestSaleDate() { return latestSaleDate; }

    private double agedSupport(Map<String, Object> record, LocalDate predictionDate) {
        double support = Math.expm1(numeric(record, "log1p_decayed_sales_count", 0.0));
        if (support <= 0.0 || asOfDate == null) return support;
        long elapsed = Math.max(0, ChronoUnit.DAYS.between(asOfDate, predictionDate));
        return support * Math.pow(0.5, elapsed / decayHalfLifeDays);
    }

    private double recency(Map<String, Object> record, LocalDate predictionDate) {
        LocalDate lastSale = record.get("_last_sale_date") instanceof LocalDate date ? date : null;
        if (lastSale == null) return maxRecencyYears;
        long days = ChronoUnit.DAYS.between(lastSale, predictionDate);
        return Math.min(Math.max(days / 365.25, 0.0), maxRecencyYears);
    }

    /**
     * Return local minus global rolling trend without repeating immutable
     * snapshot scans for every property in an ingestion batch.
     */
    private double relativeTrend(
            String cellId, Map<String, Object> cell, LocalDate predictionDate) {
        OptionalDouble local = cellTrendCache.computeIfAbsent(
            new TrendKey(cellId, predictionDate),
            ignored -> optionalTrend(windowTrend(cell, predictionDate))
        );
        if (local.isEmpty()) return 0.0;
        OptionalDouble globalTrend = globalTrendCache.computeIfAbsent(
            predictionDate,
            ignored -> optionalTrend(windowTrend(global, predictionDate))
        );
        return local.getAsDouble() - (globalTrend.isPresent() ? globalTrend.getAsDouble() : 0.0);
    }

    private static OptionalDouble optionalTrend(Double value) {
        return value == null ? OptionalDouble.empty() : OptionalDouble.of(value);
    }

    private record TrendKey(String cellId, LocalDate predictionDate) {}

    private Double windowTrend(Map<String, Object> record, LocalDate predictionDate) {
        Object rawSales = record.get("trend_sales");
        if (!(rawSales instanceof List<?> sales)) return null;
        LocalDate recentStart = predictionDate.minusDays(trendWindowDays);
        LocalDate priorStart = recentStart.minusDays(trendWindowDays);
        double recentWeight = 0.0, recentTotal = 0.0, priorWeight = 0.0, priorTotal = 0.0;
        for (Object raw : sales) {
            if (!(raw instanceof Map<?, ?> sale)) continue;
            LocalDate saleDate = sale.get("_sale_date") instanceof LocalDate date ? date : null;
            Object rawPrice = sale.get("log_price");
            if (saleDate == null || !(rawPrice instanceof Number price) || !saleDate.isBefore(predictionDate)) continue;
            double weight = Math.pow(0.5,
                ChronoUnit.DAYS.between(saleDate, predictionDate) / decayHalfLifeDays);
            if (!saleDate.isBefore(recentStart)) {
                recentWeight += weight;
                recentTotal += weight * price.doubleValue();
            } else if (!saleDate.isBefore(priorStart)) {
                priorWeight += weight;
                priorTotal += weight * price.doubleValue();
            }
        }
        return recentWeight <= 0.0 || priorWeight <= 0.0
            ? null : recentTotal / recentWeight - priorTotal / priorWeight;
    }

    @SuppressWarnings("unchecked")
    private void prepareDates(Map<String, Object> record) {
        LocalDate lastSale = parseDate(record.get("last_sale_date"));
        if (lastSale != null) record.put("_last_sale_date", lastSale);
        Object rawSales = record.get("trend_sales");
        if (!(rawSales instanceof List<?> sales)) return;
        for (Object raw : sales) {
            if (raw instanceof Map<?, ?> sale) {
                LocalDate saleDate = parseDate(sale.get("sale_date"));
                if (saleDate != null) ((Map<String, Object>) sale).put("_sale_date", saleDate);
            }
        }
    }

    private LocalDate parseDate(Object value) {
        if (!(value instanceof String text) || text.isBlank()) return null;
        return LocalDate.parse(text.substring(0, Math.min(10, text.length())));
    }

    private double numeric(Map<String, Object> record, String name, double fallback) {
        Object value = record.get(name);
        return value instanceof Number number ? number.doubleValue() : fallback;
    }

    private InputStream resource(String name) {
        InputStream input = getClass().getClassLoader().getResourceAsStream("model-artifacts/" + name);
        if (input == null) throw new IllegalStateException("Missing resource model-artifacts/" + name);
        return input;
    }
}
