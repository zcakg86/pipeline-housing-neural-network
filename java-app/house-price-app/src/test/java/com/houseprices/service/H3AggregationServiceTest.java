package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import com.uber.h3core.H3Core;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class H3AggregationServiceTest {

    @Test
    void aggregatesMatchingRecordsInOneHexAndValidatesDates() throws Exception {
        String h3 = H3Core.newInstance().latLngToCellAddress(47.6, -122.3, 8);
        PropertyStore store = new PropertyStore();
        store.upsert(List.of(
            record("1", h3, 500_000, 510_000),
            record("2", h3, 700_000, 730_000)
        ));
        H3AggregationService service = new H3AggregationService();
        service.store = store;

        PropertyRequestFilter filter = PropertyRequestFilter.parse(
            "all", "2026-01-01", "2026-12-31", -500, 500,
            null, null, null, null
        );
        List<H3AggregationService.HexStats> result = service.aggregateSales(
            "sale_price", "neural", filter
        );
        assertEquals(1, result.size());
        assertEquals(2, result.getFirst().numSales());
        assertEquals(600_000, result.getFirst().avgSalePrice(), 0.01);
        assertEquals(620_000, result.getFirst().avgNeuralPredictedPrice(), 0.01);
        assertEquals(7, result.getFirst().boundary().size());

        assertThrows(IllegalArgumentException.class, () -> PropertyRequestFilter.parse(
            "all", "bad-date", "", -500, 500, null, null, null, null
        ));
    }

    private PropertyRecord record(String id, String h3, double price, double prediction) {
        return new PropertyRecord(
            id, "address", "sales", 47.6, -122.3, h3, "1",
            2_000, 4_000, 3, 2, "Single Family", LocalDate.of(2026, 7, 1),
            price, 0, null, prediction, 100 * (prediction - price) / price,
            prediction, 100 * (prediction - price) / price, 20_000, 15,
            new float[]{.1f, .2f, .1f, .2f, .2f, .2f}
        );
    }
}
