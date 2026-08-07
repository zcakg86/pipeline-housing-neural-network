package com.houseprices.service;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TransportFeatureServiceTest {

    @Test
    void servesACompleteStaticFeatureProjectionForTheSyntheticGrid() {
        TransportFeatureService service = new TransportFeatureService();
        service.load();

        Map<String, Object> collection = service.featureCollection();
        assertEquals("FeatureCollection", collection.get("type"));
        List<?> features = (List<?>) collection.get("features");
        Map<?, ?> metadata = (Map<?, ?>) collection.get("metadata");
        assertEquals(((Number) metadata.get("cellCount")).intValue(), features.size());
        assertTrue(features.size() > 1_000);

        Map<?, ?> firstFeature = (Map<?, ?>) features.getFirst();
        Map<?, ?> properties = (Map<?, ?>) firstFeature.get("properties");
        assertTrue(properties.containsKey("light_rail_proximity"));
        assertTrue(properties.containsKey("bus_centrality_score"));
        assertTrue(properties.containsKey("major_road_density_km_per_sq_km"));
        assertTrue(((Number) properties.get("bus_centrality_score")).doubleValue() >= 0.0);
        assertTrue(((Number) properties.get("bus_centrality_score")).doubleValue() <= 1.0);
    }
}
