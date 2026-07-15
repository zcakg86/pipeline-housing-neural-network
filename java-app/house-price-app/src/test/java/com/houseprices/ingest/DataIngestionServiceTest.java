package com.houseprices.ingest;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class DataIngestionServiceTest {

    @Test
    void createsUniqueIdsForDuplicatePreferredIds() {
        String first = DataIngestionService.buildUniqueRecordId("sales", "2", 10, 47.6062, -122.3321);
        String second = DataIngestionService.buildUniqueRecordId("sales", "2", 11, 47.6062, -122.3321);

        assertEquals("sales_2_10", first);
        assertEquals("sales_2_11", second);
        assertNotEquals(first, second);
    }
}
