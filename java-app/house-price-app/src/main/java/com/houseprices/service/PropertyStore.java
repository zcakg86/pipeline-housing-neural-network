package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * In-memory store for all property records.
 * Partitioned by source for efficient querying.
 */
@ApplicationScoped
public class PropertyStore {

    private static final Logger LOG = Logger.getLogger(PropertyStore.class);

    // source -> id -> record
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, PropertyRecord>> store
        = new ConcurrentHashMap<>();
    private final AtomicLong version = new AtomicLong();

    public record Snapshot(
        long version,
        List<PropertyRecord> sales,
        List<PropertyRecord> rentcast,
        List<PropertyRecord> zillow
    ) {
        public List<PropertyRecord> completedSales() {
            List<PropertyRecord> result = new ArrayList<>(sales.size() + rentcast.size());
            result.addAll(sales);
            result.addAll(rentcast);
            return result;
        }
    }

    public void upsert(List<PropertyRecord> records) {
        boolean changed = false;
        for (PropertyRecord r : records) {
            PropertyRecord previous = store
                .computeIfAbsent(r.source(), k -> new ConcurrentHashMap<>())
                .put(r.id(), r);
            changed |= !sameRecord(previous, r);
        }
        if (changed) version.incrementAndGet();
        LOG.debugf("Upserted %d records", records.size());
    }

    private boolean sameRecord(PropertyRecord left, PropertyRecord right) {
        if (left == right) return true;
        if (left == null || right == null) return false;
        return Objects.equals(left.id(), right.id())
            && Objects.equals(left.address(), right.address())
            && Objects.equals(left.source(), right.source())
            && Double.compare(left.lat(), right.lat()) == 0
            && Double.compare(left.lng(), right.lng()) == 0
            && Objects.equals(left.h3Index(), right.h3Index())
            && Objects.equals(left.community(), right.community())
            && Double.compare(left.sqft(), right.sqft()) == 0
            && Double.compare(left.sqftLot(), right.sqftLot()) == 0
            && left.beds() == right.beds()
            && Double.compare(left.baths(), right.baths()) == 0
            && Objects.equals(left.homeType(), right.homeType())
            && Objects.equals(left.saleDate(), right.saleDate())
            && Double.compare(left.salePrice(), right.salePrice()) == 0
            && Double.compare(left.zestimate(), right.zestimate()) == 0
            && Objects.equals(left.listingUrl(), right.listingUrl())
            && Double.compare(left.predictedPrice(), right.predictedPrice()) == 0
            && Double.compare(left.pctError(), right.pctError()) == 0
            && Double.compare(left.lightgbmPredictedPrice(), right.lightgbmPredictedPrice()) == 0
            && Double.compare(left.lightgbmPctError(), right.lightgbmPctError()) == 0
            && Double.compare(left.gnnPredictedPrice(), right.gnnPredictedPrice()) == 0
            && Double.compare(left.gnnPctError(), right.gnnPctError()) == 0
            && Double.compare(left.predictionStdPrice(), right.predictionStdPrice()) == 0
            && Double.compare(left.predictionCvPct(), right.predictionCvPct()) == 0
            && Arrays.equals(left.clsAttention(), right.clsAttention());
    }

    public List<PropertyRecord> getBySource(String source) {
        Map<String, PropertyRecord> partition = store.get(source);
        if (partition == null) return List.of();
        return new ArrayList<>(partition.values());
    }

    public Optional<PropertyRecord> findBySourceAndId(String source, String id) {
        Map<String, PropertyRecord> partition = store.get(source);
        return partition == null ? Optional.empty() : Optional.ofNullable(partition.get(id));
    }

    public List<PropertyRecord> getAll() {
        return store.values().stream()
            .flatMap(m -> m.values().stream())
            .collect(Collectors.toList());
    }

    public List<PropertyRecord> getSalesRecords() {
        return snapshot().completedSales();
    }

    public List<PropertyRecord> getRentcastRecords() {
        List<PropertyRecord> result = new ArrayList<>();
        result.addAll(getBySource("rentcast"));
        return result;
    }

    public List<PropertyRecord> getZillowListings() {
        return getBySource("zillow");
    }

    public int totalCount() {
        return store.values().stream().mapToInt(Map::size).sum();
    }

    public Map<String, Integer> countsBySource() {
        Map<String, Integer> counts = new LinkedHashMap<>();
        store.forEach((src, map) -> counts.put(src, map.size()));
        return counts;
    }

    public long version() { return version.get(); }

    /** One reusable, read-only copy of every source partition for a request. */
    public Snapshot snapshot() {
        return new Snapshot(
            version.get(),
            List.copyOf(getBySource("sales")),
            List.copyOf(getBySource("rentcast")),
            List.copyOf(getBySource("zillow"))
        );
    }
}
