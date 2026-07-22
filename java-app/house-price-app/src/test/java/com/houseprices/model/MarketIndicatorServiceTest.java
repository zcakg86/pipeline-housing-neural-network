package com.houseprices.model;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MarketIndicatorServiceTest {

    @Test
    void usesOnlyIndicatorsOnOrBeforePredictionDate() {
        MarketIndicatorService service = new MarketIndicatorService();
        service.load();

        MarketIndicatorService.MarketIndicators historical =
            service.lookup(LocalDate.of(2020, 1, 1));
        assertEquals(LocalDate.of(2020, 1, 1), historical.effectiveDate());
        assertEquals(3.74, historical.mortgageRate(), 0.0001);
        assertEquals(3.6, historical.unemploymentRate(), 0.0001);

        MarketIndicatorService.MarketIndicators future =
            service.lookup(LocalDate.of(2030, 1, 1));
        assertEquals(LocalDate.of(2026, 6, 16), future.effectiveDate());
        assertEquals(6.52, future.mortgageRate(), 0.0001);
        assertEquals(4.3, future.unemploymentRate(), 0.0001);
    }
}
