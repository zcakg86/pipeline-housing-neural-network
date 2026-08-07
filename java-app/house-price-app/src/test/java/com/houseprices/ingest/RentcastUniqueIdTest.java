package com.houseprices.ingest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.math.BigInteger;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RentcastUniqueIdTest {

    @Test
    void createsDigitsOnlyIntegerWithoutLosingPrecision() {
        Map<String, Object> record = new HashMap<>();
        record.put("lastSaleDate", "2026-03-20T00:00:00Z");
        record.put("zipCode", "98106-2112");
        record.put("assessorID", "700-A0");

        RentcastUniqueId.addTo(record);

        assertInstanceOf(BigInteger.class, record.get("uniqueId"));
        assertEquals("202603209810621127000", RentcastUniqueId.storeKey(record.get("uniqueId")));
    }
}
