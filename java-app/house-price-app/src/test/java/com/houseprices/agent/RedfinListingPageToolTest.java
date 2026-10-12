package com.houseprices.agent;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.net.URI;
import java.time.Instant;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RedfinListingPageToolTest {

    @Test
    void extractsLotSizeAndNewestCompletedSaleFromScopedElements() {
        String html = """
            <div><span class="valueType">Lot Size</span><span class="valueText">6,250 Sq. Ft.</span></div>
            <div><span class="valueType">HOA Dues</span><span class="valueText">$485/month</span></div>
            <div id="property-history-transition-node">
              <div class="PropertyHistoryEventTable">
                <div class="BasicTable__row"><span>Date</span></div>
                <div class="BasicTable__row">
                  <div class="event">Sold <span>source text</span></div>
                  <div class="date">Jan 5, 2020 <span>nested</span></div>
                  <div class="price">$650,000 <span>$500/sq ft</span></div>
                </div>
                <div class="BasicTable__row">
                  <div class="event">Listed</div><div class="date">Feb 1, 2025</div>
                  <div class="price">$900,000</div>
                </div>
                <div class="BasicTable__row">
                  <div class="event">Sold</div><div class="date">Mar 12, 2024</div>
                  <div class="price">$825,500</div>
                </div>
              </div>
            </div>
            """;

        var result = RedfinListingPageTool.extractPropertyFields(
            html, "https://www.redfin.com/example", Instant.parse("2026-10-12T00:00:00Z")
        );

        assertEquals("ok", result.status());
        assertEquals("6,250 Sq. Ft.", result.lotSize());
        assertEquals("$485/month", result.hoaDues());
        assertEquals(LocalDate.of(2024, 3, 12), result.lastSaleDate());
        assertEquals(new BigDecimal("825500"), result.lastSalePrice());
    }

    @Test
    void reportsAnUnloadedHistoryTableAndNormalizesPlaceholderLotSize() {
        var result = RedfinListingPageTool.extractPropertyFields(
            "<div><span class='valueType'>lot size</span><span class='valueText'>—</span></div>" +
                "<div><span class='valueType'>hoa dues</span><span class='valueText'>--</span></div>",
            "https://www.redfin.com/example", Instant.EPOCH
        );

        assertEquals("page_incomplete", result.status());
        assertNull(result.lotSize());
        assertNull(result.hoaDues());
        assertNull(result.lastSaleDate());
        assertTrue(result.note().contains("missing"));
    }

    @Test
    void permitsOnlyHttpsRedfinHosts() {
        assertTrue(RedfinListingPageTool.isAllowedRedfinUri(
            URI.create("https://www.redfin.com/WA/Seattle/home/1")
        ));
        assertTrue(RedfinListingPageTool.isAllowedRedfinUri(
            URI.create("https://redfin.com/home/1")
        ));
        assertFalse(RedfinListingPageTool.isAllowedRedfinUri(
            URI.create("http://www.redfin.com/home/1")
        ));
        assertFalse(RedfinListingPageTool.isAllowedRedfinUri(
            URI.create("https://redfin.com.example.org/home/1")
        ));
        assertFalse(RedfinListingPageTool.isAllowedRedfinUri(
            URI.create("https://example.org/home/1")
        ));
        assertFalse(RedfinListingPageTool.isAllowedRedfinUri(
            URI.create("https://www.redfin.com:8443/home/1")
        ));
    }
}
