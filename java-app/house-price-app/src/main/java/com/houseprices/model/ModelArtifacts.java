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
    private HashMap<String, Integer> communityVocab = new HashMap<>();
    private HashMap<String, Integer> yearVocab      = new HashMap<>();
    private HashMap<String, Integer> weekVocab      = new HashMap<>();

    // H3 L9 -> [7 community indices]
    private HashMap<String, int[]> h3NeighborMap = new HashMap<>();

    // H3 L7 -> community ID
    private HashMap<String, String> communityMap = new HashMap<>();

    // Metadata
    private LocalDate referenceDate;
    private int unknownCommunityIdx;
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

            // ── Vocabularies ─────────────────────────────────────────────────
            communityVocab = mapper.readValue(resource("community_vocab.json"),
                mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));
            yearVocab = mapper.readValue(resource("year_vocab.json"),
                mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));
            weekVocab = mapper.readValue(resource("week_vocab.json"),
                mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));

            unknownCommunityIdx = communityVocab.getOrDefault("unknown", communityVocab.size() - 1);
            unknownYearIdx      = yearVocab.getOrDefault("unknown", yearVocab.size() - 1);
            unknownWeekIdx      = weekVocab.getOrDefault("unknown", weekVocab.size() - 1);

            LOG.infof("Loaded vocabs: community=%d, year=%d, week=%d",
                communityVocab.size(), yearVocab.size(), weekVocab.size());

            // ── H3 Neighbor Map ───────────────────────────────────────────────
            Map<String, List<Integer>> rawH3 = mapper.readValue(
                resource("h3_l9_neighbor_communities.json"),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, List.class)
            );
            rawH3.forEach((hex, neighbors) -> {
                int[] arr = neighbors.stream().mapToInt(Integer::intValue).toArray();
                h3NeighborMap.put(hex, arr);
            });
            LOG.infof("Loaded H3 neighbor map: %d hexes", h3NeighborMap.size());

            // ── Community map (H3 L9 → community ID) ─────────────────────────
            // Bundled in model-artifacts as h3_l9_to_community.json
            InputStream commIs = getClass().getClassLoader()
                .getResourceAsStream("model-artifacts/h3_l9_to_community.json");
            if (commIs != null) {
                Map<String, Integer> rawCommunity = mapper.readValue(commIs,
                    mapper.getTypeFactory().constructMapType(HashMap.class, String.class, Integer.class));
                rawCommunity.forEach((k, v) -> communityMap.put(k, String.valueOf(v)));
                LOG.infof("Loaded H3 L9 community map: %d entries", communityMap.size());
            } else {
                LOG.warn("h3_l9_to_community.json not found in model-artifacts");
            }

            // ── Metadata ──────────────────────────────────────────────────────
            Map<String, Object> meta = mapper.readValue(resource("model_metadata.json"),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
            String refDateStr = (String) meta.get("reference_date");
            if (refDateStr != null) {
                referenceDate = LocalDate.parse(refDateStr.substring(0, 10));
            }
            LOG.infof("Reference date: %s", referenceDate);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load model artifacts", e);
        }
    }

    // ── Scaler helpers ────────────────────────────────────────────────────────

    public double scaleFeature(String feature, double value) {
        double[] ms = scalers.get(feature);
        if (ms == null) throw new IllegalArgumentException("Unknown scaler: " + feature);
        return (value - ms[0]) / ms[1];
    }

    public double inverseScaleLogPrice(double scaledLogPrice) {
        double[] ms = scalers.get("log_price");
        return scaledLogPrice * ms[1] + ms[0];
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

    /** Resolve H3 L9 index to community ID string, empty string if unknown */
    public String lookupCommunity(String h3L9) {
        if (h3L9 == null || h3L9.isBlank()) return "";
        return communityMap.getOrDefault(h3L9, "");
    }

    private InputStream resource(String name) {
        InputStream is = getClass().getClassLoader().getResourceAsStream("model-artifacts/" + name);
        if (is == null) throw new RuntimeException("Resource not found: model-artifacts/" + name);
        return is;
    }
}
