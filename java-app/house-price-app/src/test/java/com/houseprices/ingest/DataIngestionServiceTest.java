package com.houseprices.ingest;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DataIngestionServiceTest {

    @Test
    void createsDeterministicUniqueIdsFromSaleDateAndCsvRow() {
        String first = DataIngestionService.buildUniqueRecordId(LocalDate.of(2020, 6, 29), 87407);
        String second = DataIngestionService.buildUniqueRecordId(LocalDate.of(2020, 6, 29), 87408);

        assertEquals("2020062987407", first);
        assertEquals("2020062987408", second);
        assertNotEquals(first, second);
    }

    @Test
    void usesLivingAreaWhenListingLotAreaIsMissing() {
        DataIngestionService service = new DataIngestionService();

        assertEquals(1_850.0, service.listingLotSqft(Map.of(), 1_850.0));
        assertEquals(9_600.0, service.listingLotSqft(
            Map.of("lotAreaValue", 9_600), 1_850.0));

        Map<String, Object> nullLot = new HashMap<>();
        nullLot.put("lotAreaValue", null);
        assertEquals(1_850.0, service.listingLotSqft(nullLot, 1_850.0));
    }

    @Test
    void choosesListingLotFallbackFromHomeType() {
        DataIngestionService service = new DataIngestionService();

        assertEquals(3_000.0, service.listingLotFallback("House", 2_000.0));
        assertEquals(2_000.0, service.listingLotFallback("Condo", 2_000.0));
        assertEquals(2_000.0, service.listingLotFallback("Townhouse", 2_000.0));
        assertEquals(2_000.0, service.listingLotFallback("", 2_000.0));
    }

    @Test
    void parsesDisplayedRedfinLotSizesIntoSquareFeet() {
        assertEquals(6_250.0, DataIngestionService.parseLotSizeSqft("6,250 Sq. Ft."));
        assertEquals(10_890.0, DataIngestionService.parseLotSizeSqft("0.25 Acres"));
        assertEquals(7_405.0, DataIngestionService.parseLotSizeSqft("7,405 sqft"));
        assertTrue(Double.isNaN(DataIngestionService.parseLotSizeSqft("—")));
    }

    @Test
    void prefersRetrievedRedfinLotSizeButKeepsSuppliedApiValue() {
        DataIngestionService service = new DataIngestionService();
        Map<String, Double> retrieved = Map.of("https://www.redfin.com/home/1", 6_250.0);

        assertEquals(6_250.0, service.listingLotSqft(
            Map.of(), "redfin", "https://www.redfin.com/home/1", 2_000.0, retrieved
        ));
        assertEquals(9_600.0, service.listingLotSqft(
            Map.of("lotAreaValue", 9_600), "redfin",
            "https://www.redfin.com/home/1", 2_000.0, retrieved
        ));
        assertEquals(2_000.0, service.listingLotSqft(
            Map.of(), "zillow", "https://www.redfin.com/home/1", 2_000.0, retrieved
        ));
    }
}
