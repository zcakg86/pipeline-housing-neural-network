package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PropertyStoreTest {

    @Test
    void snapshotCombinesSalesOnceAndVersionChangesOnlyForChangedRecords() {
        PropertyStore store = new PropertyStore();
        PropertyRecord sale = record("1", "sales", 500_000);
        PropertyRecord rentcast = record("2", "rentcast", 600_000);

        store.upsert(List.of(sale, rentcast));
        long version = store.version();
        PropertyStore.Snapshot snapshot = store.snapshot();
        assertEquals(2, snapshot.completedSales().size());

        store.upsert(List.of(sale));
        assertEquals(version, store.version());
        store.upsert(List.of(record("1", "sales", 510_000)));
        assertEquals(version + 1, store.version());
    }

    @Test
    void retainsRedfinDescriptionAndAtAGlanceFacts() {
        PropertyStore store = new PropertyStore();
        PropertyRecord base = record("redfin-1", "redfin", 700_000);
        PropertyRecord redfin = new PropertyRecord(
            base.id(), base.address(), base.source(), base.lat(), base.lng(), base.h3Index(),
            base.community(), base.sqft(), base.sqftLot(), base.beds(), base.baths(),
            base.homeType(), base.saleDate(), base.salePrice(), base.zestimate(),
            "https://www.redfin.com/home/1", base.predictedPrice(), base.pctError(),
            base.lightgbmPredictedPrice(), base.lightgbmPctError(), base.gnnPredictedPrice(),
            base.gnnPctError(), base.predictionStdPrice(), base.predictionCvPct(),
            base.clsAttention(), "A useful listing description", List.of("Open kitchen", "Garage")
        );
        store.upsert(List.of(redfin));

        PropertyRecord stored = store.getRedfinListings().getFirst();
        assertEquals("A useful listing description", stored.description());
        assertEquals(List.of("Open kitchen", "Garage"), stored.atAGlanceFacts());
        assertTrue(store.snapshot().redfin().contains(stored));
    }

    private PropertyRecord record(String id, String source, double price) {
        return new PropertyRecord(
            id, "address", source, 47.6, -122.3, "8828d542d7fffff", "1",
            2_000, 4_000, 3, 2, "Single Family", LocalDate.of(2026, 7, 1),
            price, 0, null, 520_000, 4, 515_000, 3, 20_000, 15,
            510_000, 2,
            new float[]{.1f, .3f, .3f, .3f}, "", List.of()
        );
    }
}
