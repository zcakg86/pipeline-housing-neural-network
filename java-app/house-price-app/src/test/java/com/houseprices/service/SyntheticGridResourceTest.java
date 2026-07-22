package com.houseprices.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.uber.h3core.H3Core;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SyntheticGridResourceTest {

    @Test
    void packagesEveryLevelEightCellWithCentroidAndClosedBoundary() throws Exception {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(
                "model-artifacts/synthetic_h3_l8_grid.json")) {
            assertNotNull(input);
            JsonNode grid = new ObjectMapper().readTree(input);
            JsonNode cells = grid.path("cells");
            assertEquals(grid.path("cell_count").asInt(), cells.size());
            assertEquals(2_947, cells.size());

            H3Core h3 = H3Core.newInstance();
            Set<String> indexes = new HashSet<>();
            for (JsonNode cell : cells) {
                String h3Index = cell.path("h3_l8").asText();
                assertTrue(indexes.add(h3Index));
                assertEquals(8, h3.getResolution(h3.stringToH3(h3Index)));
                assertTrue(cell.path("lat").isNumber());
                assertTrue(cell.path("lng").isNumber());
                JsonNode boundary = cell.path("boundary");
                assertTrue(boundary.size() >= 6);
                assertEquals(boundary.get(0), boundary.get(boundary.size() - 1));
            }
        }
    }
}
