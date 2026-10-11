package com.houseprices.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Lightweight, precomputed H3 place and historical-market context for the agent. */
@ApplicationScoped
public class H3MarketSurfaceService {

    private static final Logger LOG = Logger.getLogger(H3MarketSurfaceService.class);

    @Inject ObjectMapper mapper;

    private Map<String, Map<String, Object>> cells = Map.of();
    private Map<String, Object> metadata = Map.of();

    @PostConstruct
    @SuppressWarnings("unchecked")
    void load() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(
                "model-artifacts/h3_market_surface.json")) {
            if (input == null) throw new IllegalStateException("h3_market_surface.json not found");
            Map<String, Object> payload = mapper.readValue(input, Map.class);
            Object rawCells = payload.remove("cells");
            if (!(rawCells instanceof Map<?, ?> values)) {
                throw new IllegalStateException("h3_market_surface.json has no cells object");
            }
            Map<String, Map<String, Object>> loaded = new LinkedHashMap<>();
            values.forEach((key, value) -> {
                if (value instanceof Map<?, ?> row) {
                    loaded.put(String.valueOf(key), (Map<String, Object>) row);
                }
            });
            cells = Map.copyOf(loaded);
            metadata = Collections.unmodifiableMap(new LinkedHashMap<>(payload));
            LOG.infof("Loaded lightweight H3 market surface: %,d cells", cells.size());
        } catch (Exception exception) {
            throw new RuntimeException("Failed to load H3 market surface", exception);
        }
    }

    public Map<String, Object> lookup(String h3Index) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("h3Index", h3Index);
        // Map.copyOf produces an immutable map whose get(null) throws. Missing
        // model-neighbor slots are intentionally represented by null cell IDs.
        Map<String, Object> value = h3Index == null ? null : cells.get(h3Index);
        if (value != null) result.putAll(value);
        result.put("available", value != null);
        return result;
    }

    public List<Map<String, Object>> lookupAlignedNeighborhood(String[] h3Indexes) {
        List<Map<String, Object>> result = new ArrayList<>(h3Indexes.length);
        for (int index = 0; index < h3Indexes.length; index++) {
            String cell = h3Indexes[index];
            Map<String, Object> row = lookup(cell);
            row.put("relation", index == 0 ? "selected cell" : "model neighbor " + index);
            result.add(row);
        }
        return result;
    }

    public Map<String, Object> metadata() {
        return metadata;
    }
}
