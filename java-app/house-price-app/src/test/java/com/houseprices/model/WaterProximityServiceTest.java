package com.houseprices.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WaterProximityServiceTest {

    @Test
    void calculatesDistanceAndExponentialProximity() {
        WaterProximityService service = new WaterProximityService();
        service.load();

        WaterProximityService.WaterFeatures near = service.lookup(47.6205, -122.3493);
        WaterProximityService.WaterFeatures inland = service.lookup(47.6097, -122.3331);

        assertTrue(near.distanceToWaterM() < 50.0);
        assertTrue(inland.distanceToWaterM() > 50.0);
        assertEquals(1.0, WaterProximityService.waterProximity(0.0), 1e-12);
        assertEquals(Math.exp(-5.0), WaterProximityService.waterProximity(500.0), 1e-12);
    }
}
