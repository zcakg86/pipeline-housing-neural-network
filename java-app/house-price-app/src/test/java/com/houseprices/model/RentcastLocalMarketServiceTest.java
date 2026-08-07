package com.houseprices.model;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RentcastLocalMarketServiceTest {

    @Test
    void loadsRollingMixedSourceStateAndMarksHistoricalRentcastPredictions() {
        RentcastLocalMarketService service = new RentcastLocalMarketService();
        service.load();

        assertEquals(LocalDate.of(2026, 3, 30), service.latestSaleDate());
        assertTrue(service.hasLookAheadIssue(LocalDate.of(2026, 3, 30)));
        assertTrue(!service.hasLookAheadIssue(LocalDate.of(2026, 3, 31)));
        float[][] values = service.rawFeatures("8828d542b3fffff", LocalDate.of(2026, 7, 1));
        assertEquals(7, values.length);
        assertEquals(5, values[0].length);
    }
}
