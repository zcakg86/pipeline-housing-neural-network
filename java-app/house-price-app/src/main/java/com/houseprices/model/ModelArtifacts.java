package com.houseprices.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.annotation.PostConstruct;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.time.LocalDate;
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

    // H3 L9 -> community ID (integer, stored as String for display)
    // Loaded from community_map.json (the source of truth — no separate vocab file needed)
    private HashMap<String, Integer> communityMap = new HashMap<>();

    // Metadata
    private LocalDate referenceDate;
    private int unknownCommunityIdx;   // = n_communities from model_metadata.json
    private int unknownYearIdx;
    private int unknownWeekIdx;

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

    public LocalDate getReferenceDate() { return referenceDate; }
    public int getUnknownCommunityIdx() { return unknownCommunityIdx; }

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
}
