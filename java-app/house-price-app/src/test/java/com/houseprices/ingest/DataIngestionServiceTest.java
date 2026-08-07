package com.houseprices.ingest;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class DataIngestionServiceTest {

    @Test
    void createsDeterministicUniqueIdsFromSaleDateAndCsvRow() {
        String first = DataIngestionService.buildUniqueRecordId(LocalDate.of(2020, 6, 29), 87407);
        String second = DataIngestionService.buildUniqueRecordId(LocalDate.of(2020, 6, 29), 87408);

        assertEquals("2020062987407", first);
        assertEquals("2020062987408", second);
        assertNotEquals(first, second);
    }
}
