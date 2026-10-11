package com.houseprices.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class H3MarketSurfaceServiceTest {

    @Test
    void loadsCompactHistoricalAndPlaceEvidence() {
        H3MarketSurfaceService service = new H3MarketSurfaceService();
        service.mapper = new ObjectMapper();
        service.load();

        Map<String, Object> bellevue = service.lookup("8828d54309fffff");
        assertEquals(true, bellevue.get("available"));
        assertEquals("Bellevue", ((Map<?, ?>) bellevue.get("city")).get("name"));
        assertNotNull(bellevue.get("neighborhood"));
        Map<?, ?> market = (Map<?, ?>) bellevue.get("historicalMarket");
        assertTrue(((Number) market.get("decayedSaleSupport")).doubleValue() > 0.0);
        assertTrue(((Number) market.get("historicalTypicalSalePriceAllProperties")).doubleValue() > 0.0);

        List<Map<String, Object>> aligned = service.lookupAlignedNeighborhood(new String[] {
            "8828d54309fffff", "not-a-covered-cell", null
        });
        assertEquals("selected cell", aligned.get(0).get("relation"));
        assertEquals("model neighbor 1", aligned.get(1).get("relation"));
        assertEquals(false, aligned.get(1).get("available"));
        assertEquals("model neighbor 2", aligned.get(2).get("relation"));
        assertEquals(false, aligned.get(2).get("available"));
        assertFalse(service.metadata().containsKey("cells"));
        assertEquals(8, service.metadata().get("h3Resolution"));
    }
}
