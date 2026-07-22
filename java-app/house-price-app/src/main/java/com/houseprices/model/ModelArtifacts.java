package com.houseprices.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.annotation.PostConstruct;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads and holds all model artifacts: scalers, vocabularies, H3 neighbor map, metadata.
 * All vocabs stored as HashMap<String, Integer> for O(1) lookup.
 */
@ApplicationScoped
public class ModelArtifacts {

    private static final Logger LOG = Logger.getLogger(ModelArtifacts.class);

    // Scalers: feature -> {mean, scale}
    private Map<String, double[]> scalers = new HashMap<>();

    // Vocabularies: value -> index
    private HashMap<String, Integer> yearVocab = new HashMap<>();
    private HashMap<String, Integer> weekVocab = new HashMap<>();

    // H3 L9 -> [7 community indices]
    private HashMap<String, int[]> h3NeighborMap = new HashMap<>();
    private HashMap<String, String[]> h3NeighborCells = new HashMap<>();
    private Map<String, Map<String, Object>> localMarketCells = new HashMap<>();
    private Map<String, Object> globalLocalMarket = new HashMap<>();
    private double maxLocalRecencyYears = 5.0;
    private int localRecentWindowDays = 365;
    private LocalDate localMarketAsOfDate;
    private LocalDate localMarketLatestSaleDate;

    // H3 L9 -> community ID (integer, stored as String for display)
    // Loaded from community_map.json (the source of truth — no separate vocab file needed)
    private HashMap<String, Integer> communityMap = new HashMap<>();

    // Metadata
    private LocalDate referenceDate;
    private int unknownCommunityIdx;   // = n_communities from model_metadata.json
    private int unknownYearIdx;
    private int unknownWeekIdx;
    private List<String> propertyFeatures = List.of("sqft", "sqft_lot", "beds");

    @PostConstruct
    void load() {
        ObjectMapper mapper = new ObjectMapper();
        try {
            // ── Scalers ──────────────────────────────────────────────────────
            Map<String, Map<String, Double>> rawScalers = mapper.readValue(
                resource("scalers.json"),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, Map.class)
            );
            rawScalers.forEach((feat, vals) ->
                scalers.put(feat, new double[]{vals.get("mean"), vals.get("scale")})
            );
            LOG.infof("Loaded %d scalers", scalers.size());

            // ── Year / week vocabularies ──────────────────────────────────────
            yearVocab = mapper.readValue(resource("year_vocab.json"),
                mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));
            weekVocab = mapper.readValue(resource("week_vocab.json"),
                mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));

            unknownYearIdx = yearVocab.getOrDefault("unknown", yearVocab.size() - 1);
            unknownWeekIdx = weekVocab.getOrDefault("unknown", weekVocab.size() - 1);

            LOG.infof("Loaded vocabs: year=%d, week=%d", yearVocab.size(), weekVocab.size());

            // ── H3 Neighbor Map ───────────────────────────────────────────────
            Map<String, List<Integer>> rawH3 = mapper.readValue(
                resource("h3_l8_neighbor_communities.json"),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, List.class)
            );
            rawH3.forEach((hex, neighbors) -> {
                int[] arr = neighbors.stream().mapToInt(Integer::intValue).toArray();
                h3NeighborMap.put(hex, arr);
            });
            LOG.infof("Loaded H3 neighbor map: %d hexes", h3NeighborMap.size());

            InputStream neighborCellsStream = optionalResource("h3_l8_neighbor_cells.json");
            if (neighborCellsStream != null) {
                Map<String, List<String>> rawCells = mapper.readValue(
                    neighborCellsStream,
                    mapper.getTypeFactory().constructMapType(Map.class, String.class, List.class)
                );
                rawCells.forEach((hex, neighbors) ->
                    h3NeighborCells.put(hex, neighbors.toArray(new String[0]))
                );
            }

            InputStream localSnapshotStream = optionalResource("local_market_snapshot.json");
            if (localSnapshotStream != null) {
                Map<String, Object> snapshot = mapper.readValue(localSnapshotStream, Map.class);
                Object rawLocalCells = snapshot.get("cells");
                if (rawLocalCells instanceof Map<?, ?> cells) {
                    cells.forEach((key, value) -> {
                        if (value instanceof Map<?, ?> record) {
                            Map<String, Object> prepared = (Map<String, Object>) record;
                            prepareLocalDates(prepared);
                            localMarketCells.put(String.valueOf(key), prepared);
                        }
                    });
                }
                Object rawGlobal = snapshot.get("global");
                if (rawGlobal instanceof Map<?, ?> record) {
                    globalLocalMarket = (Map<String, Object>) record;
                    prepareLocalDates(globalLocalMarket);
                }
                Object maxRecency = snapshot.get("max_recency_years");
                if (maxRecency instanceof Number number) {
                    maxLocalRecencyYears = number.doubleValue();
                }
                Object recentWindow = snapshot.get("recent_window_days");
                if (recentWindow instanceof Number number) {
                    localRecentWindowDays = number.intValue();
                }
                localMarketAsOfDate = parseOptionalDate(snapshot.get("as_of_date"));
                localMarketLatestSaleDate = parseOptionalDate(snapshot.get("latest_sale_date"));
                LOG.infof(
                    "Loaded local market snapshot: %d H3 cells, latest sale=%s, usable from=%s",
                    localMarketCells.size(), localMarketLatestSaleDate, localMarketAsOfDate
                );
            }

            // ── Community map (H3 L8 → community ID) ─────────────────────────
            // community_map.json is the source of truth: h3_08_hex -> community_id (int).
            // No separate community vocab file — indices are already 0-based in the neighbor map.
            communityMap = mapper.readValue(resource("community_map.json"),
                mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));
            LOG.infof("Loaded community map: %d H3 L8 entries", communityMap.size());

            // ── Metadata ──────────────────────────────────────────────────────
            Map<String, Object> meta = mapper.readValue(resource("model_metadata.json"),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
            String refDateStr = (String) meta.get("reference_date");
            if (refDateStr != null) {
                referenceDate = LocalDate.parse(refDateStr.substring(0, 10));
            }
            // n_communities is the unknown/padding index (one past the last real community)
            unknownCommunityIdx = ((Number) meta.get("n_communities")).intValue();
            Object rawPropertyFeatures = meta.get("property_features");
            if (rawPropertyFeatures instanceof List<?> names) {
                propertyFeatures = names.stream().map(String::valueOf).toList();
            }

            LOG.infof("Reference date: %s  unknownCommunityIdx: %d",
                referenceDate, unknownCommunityIdx);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load model artifacts", e);
        }
    }

    // ── Scaler helpers ────────────────────────────────────────────────────────

    public double scaleFeature(String feature, double value) {
        double[] ms = scalers.get(feature);
        if (ms == null) {
            LOG.warnf("Unknown scaler '%s' — returning raw value. Update model artifacts.", feature);
            return value;  // return unscaled rather than throwing, to avoid silently swallowing records
        }
        return (value - ms[0]) / ms[1];
    }

    public double inverseScaleLogPrice(double scaledLogPrice) {
        double[] ms = scalers.get("log_price");
        return scaledLogPrice * ms[1] + ms[0];
    }

    /** The scale_ parameter of the log_price scaler (std dev of log prices in training data). */
    public double getLogPriceScale() {
        double[] ms = scalers.get("log_price");
        return ms != null ? ms[1] : 1.0;
    }

    // ── Vocab helpers ─────────────────────────────────────────────────────────

    public int lookupYear(int year) {
        return yearVocab.getOrDefault(String.valueOf(year), unknownYearIdx);
    }

    public int lookupWeek(int week) {
        return weekVocab.getOrDefault(String.valueOf(week), unknownWeekIdx);
    }

    public int[] lookupH3Neighbors(String h3Index) {
        int[] neighbors = h3NeighborMap.get(h3Index);
        if (neighbors != null) return neighbors;
        // Return all-unknown if hex not in map
        int[] unknown = new int[7];
        java.util.Arrays.fill(unknown, unknownCommunityIdx);
        return unknown;
    }

    /** Build scaled [7,5] center-plus-neighbor market features as of saleDate. */
    public float[][] lookupLocalMarketFeatures(String h3Index, LocalDate saleDate) {
        return lookupLocalMarketFeatures(h3Index, saleDate, true, true);
    }

    /** Build raw [7,5] features for tree models, which do not use NN scalers. */
    public float[][] lookupRawLocalMarketFeatures(String h3Index, LocalDate saleDate) {
        return lookupLocalMarketFeatures(h3Index, saleDate, false, true);
    }

    /** Demonstration-only lookup that intentionally permits snapshot look-ahead. */
    public float[][] lookupDemoLocalMarketFeatures(String h3Index, LocalDate saleDate) {
        return lookupLocalMarketFeatures(h3Index, saleDate, true, false);
    }

    /** Raw demonstration-only lookup that intentionally permits snapshot look-ahead. */
    public float[][] lookupDemoRawLocalMarketFeatures(String h3Index, LocalDate saleDate) {
        return lookupLocalMarketFeatures(h3Index, saleDate, false, false);
    }

    private float[][] lookupLocalMarketFeatures(
        String h3Index, LocalDate saleDate, boolean scaled, boolean enforceSnapshotDate
    ) {
        if (saleDate == null) {
            throw new IllegalArgumentException("Prediction sale date is required");
        }
        if (enforceSnapshotDate && localMarketLatestSaleDate != null
                && !saleDate.isAfter(localMarketLatestSaleDate)) {
            throw new IllegalArgumentException(
                "Prediction date " + saleDate + " must be after the local market " +
                "snapshot's latest included sale " + localMarketLatestSaleDate
            );
        }
        String[] cells = h3NeighborCells.get(h3Index);
        if (cells == null || cells.length != 7) {
            cells = new String[7];
            cells[0] = h3Index;
        }

        float[][] result = new float[7][5];
        for (int i = 0; i < 7; i++) {
            Map<String, Object> record = cells[i] == null ? null : localMarketCells.get(cells[i]);
            if (record == null) record = globalLocalMarket;

            double mean = numeric(record, "mean_log_price", Math.log(300_000.0));
            double std = numeric(record, "log_price_std", 0.5);
            double count = numeric(record, "log1p_sales_count", 0.0);
            double trend = rollingTrend(
                record,
                mean,
                saleDate,
                numeric(record, "price_trend", 0.0)
            );
            double recency = maxLocalRecencyYears;
            Object lastDateValue = record.get("_parsed_last_sale_date");
            if (lastDateValue instanceof LocalDate lastDate) {
                long days = ChronoUnit.DAYS.between(lastDate, saleDate);
                recency = Math.min(Math.max(days / 365.25, 0.0), maxLocalRecencyYears);
            }

            result[i][0] = (float) (scaled ? scaleFeature("local_mean_log_price", mean) : mean);
            result[i][1] = (float) (scaled ? scaleFeature("local_log_price_std", std) : std);
            result[i][2] = (float) (scaled ? scaleFeature("local_log1p_sales_count", count) : count);
            result[i][3] = (float) (scaled ? scaleFeature("local_recency_years", recency) : recency);
            result[i][4] = (float) (scaled ? scaleFeature("local_price_trend", trend) : trend);
        }
        return result;
    }

    private double rollingTrend(
        Map<String, Object> record,
        double longTermMean,
        LocalDate predictionDate,
        double fallback
    ) {
        if (record == null) return fallback;
        Object rawRecentSales = record.get("recent_sales");
        if (!(rawRecentSales instanceof List<?> recentSales)) return fallback;

        LocalDate cutoff = predictionDate.minusDays(localRecentWindowDays);
        double total = 0.0;
        int count = 0;
        for (Object rawSale : recentSales) {
            if (!(rawSale instanceof Map<?, ?> sale)) continue;
            Object parsedDate = sale.get("_parsed_sale_date");
            LocalDate transactionDate = parsedDate instanceof LocalDate date
                ? date : parseOptionalDate(sale.get("sale_date"));
            Object rawLogPrice = sale.get("log_price");
            if (transactionDate == null || !(rawLogPrice instanceof Number logPrice)) continue;
            if (!transactionDate.isBefore(cutoff) && transactionDate.isBefore(predictionDate)) {
                total += logPrice.doubleValue();
                count++;
            }
        }
        return count == 0 ? 0.0 : total / count - longTermMean;
    }

    private LocalDate parseOptionalDate(Object value) {
        if (!(value instanceof String text) || text.isBlank()) return null;
        return LocalDate.parse(text.substring(0, Math.min(text.length(), 10)));
    }

    @SuppressWarnings("unchecked")
    private void prepareLocalDates(Map<String, Object> record) {
        LocalDate lastSaleDate = parseOptionalDate(record.get("last_sale_date"));
        if (lastSaleDate != null) record.put("_parsed_last_sale_date", lastSaleDate);
        Object recentValue = record.get("recent_sales");
        if (!(recentValue instanceof List<?> recentSales)) return;
        for (Object rawSale : recentSales) {
            if (!(rawSale instanceof Map<?, ?> sale)) continue;
            LocalDate saleDate = parseOptionalDate(sale.get("sale_date"));
            if (saleDate != null) {
                ((Map<String, Object>) sale).put("_parsed_sale_date", saleDate);
            }
        }
    }

    private double numeric(Map<String, Object> record, String key, double fallback) {
        if (record == null) return fallback;
        Object value = record.get(key);
        return value instanceof Number number ? number.doubleValue() : fallback;
    }

    public LocalDate getReferenceDate() { return referenceDate; }
    public LocalDate getLocalMarketAsOfDate() { return localMarketAsOfDate; }
    public LocalDate getLocalMarketLatestSaleDate() { return localMarketLatestSaleDate; }
    public int getUnknownCommunityIdx() { return unknownCommunityIdx; }
    public int getUnknownYearIdx() { return unknownYearIdx; }
    public int getUnknownWeekIdx() { return unknownWeekIdx; }
    public List<String> getPropertyFeatures() { return propertyFeatures; }

    /** Resolve H3 L9 index to community ID, or empty string if not in the map */
    public String lookupCommunity(String h3L9) {
        if (h3L9 == null || h3L9.isBlank()) return "";
        Integer id = communityMap.get(h3L9);
        return id != null ? String.valueOf(id) : "";
    }

    private InputStream resource(String name) {
        InputStream is = getClass().getClassLoader().getResourceAsStream("model-artifacts/" + name);
        if (is == null) throw new RuntimeException("Resource not found: model-artifacts/" + name);
        return is;
    }

    private InputStream optionalResource(String name) {
        return getClass().getClassLoader().getResourceAsStream("model-artifacts/" + name);
    }
}
