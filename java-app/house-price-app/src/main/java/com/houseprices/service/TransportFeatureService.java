package com.houseprices.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Serves static OSM-derived accessibility fields on the synthetic H3 level-8
 * grid. These fields are map/GNN inputs only: the deployed price models do not
 * yet consume them.
 */
@ApplicationScoped
public class TransportFeatureService {

    private static final Logger LOG = Logger.getLogger(TransportFeatureService.class);
    private static final String GRID_RESOURCE = "model-artifacts/synthetic_h3_l8_grid.json";
    private static final String FEATURES_RESOURCE =
        "model-artifacts/h3_l8_transport_features.json";

    private Map<String, Object> featureCollection;

    @JsonIgnoreProperties(ignoreUnknown = true)
    private record GridFile(
        @JsonProperty("cell_count") int cellCount,
        List<GridCell> cells
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    private record GridCell(
        @JsonProperty("h3_l8") String h3L8,
        List<List<Double>> boundary
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    private record FeatureFile(
        @JsonProperty("schema_version") int schemaVersion,
        @JsonProperty("feature_names") List<String> featureNames,
        Map<String, Object> definitions,
        Map<String, Object> parameters,
        Map<String, Object> source,
        @JsonProperty("cell_count") int cellCount,
        Map<String, Map<String, Double>> cells
    ) {}

    /** Load the compact generated artifact once rather than recalculating OSM geometry in Java. */
    @PostConstruct
    void load() {
        ObjectMapper mapper = new ObjectMapper();
        try (
            InputStream gridInput = resource(GRID_RESOURCE);
            InputStream featuresInput = resource(FEATURES_RESOURCE)
        ) {
            GridFile grid = mapper.readValue(gridInput, GridFile.class);
            FeatureFile featureFile = mapper.readValue(featuresInput, FeatureFile.class);
            if (grid.cells() == null || grid.cells().size() != grid.cellCount()) {
                throw new IllegalStateException("Synthetic H3 grid count does not match its metadata");
            }
            if (featureFile.cells() == null || featureFile.cells().size() != featureFile.cellCount()) {
                throw new IllegalStateException("Transport feature count does not match its metadata");
            }

            List<Map<String, Object>> features = new ArrayList<>(grid.cells().size());
            for (GridCell cell : grid.cells()) {
                Map<String, Double> values = featureFile.cells().get(cell.h3L8());
                if (values == null) {
                    throw new IllegalStateException(
                        "No transport features for synthetic H3 cell " + cell.h3L8()
                    );
                }
                Map<String, Object> properties = new LinkedHashMap<>();
                properties.put("h3Index", cell.h3L8());
                properties.putAll(values);
                features.add(Map.of(
                    "type", "Feature",
                    "geometry", Map.of("type", "Polygon", "coordinates", List.of(cell.boundary())),
                    "properties", properties
                ));
            }
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("schemaVersion", featureFile.schemaVersion());
            metadata.put("featureNames", featureFile.featureNames());
            metadata.put("definitions", featureFile.definitions());
            metadata.put("parameters", featureFile.parameters());
            metadata.put("source", featureFile.source());
            metadata.put("cellCount", features.size());
            featureCollection = Map.of(
                "type", "FeatureCollection",
                "metadata", metadata,
                "features", List.copyOf(features)
            );
            LOG.infof("Loaded %,d static H3 transport feature records", features.size());
        } catch (Exception exception) {
            throw new RuntimeException("Failed to load H3 transport feature artifact", exception);
        }
    }

    private InputStream resource(String name) {
        InputStream input = getClass().getClassLoader().getResourceAsStream(name);
        if (input == null) throw new IllegalStateException(name + " not found in resources");
        return input;
    }

    /** A cached static GeoJSON projection of the versioned H3 feature artifact. */
    public Map<String, Object> featureCollection() {
        return featureCollection;
    }
}
