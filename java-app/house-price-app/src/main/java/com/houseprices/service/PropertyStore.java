package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
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

    public void upsert(List<PropertyRecord> records) {
        for (PropertyRecord r : records) {
            store.computeIfAbsent(r.source(), k -> new ConcurrentHashMap<>())
                 .put(r.id(), r);
        }
        LOG.debugf("Upserted %d records", records.size());
    }

    public List<PropertyRecord> getBySource(String source) {
        Map<String, PropertyRecord> partition = store.get(source);
        if (partition == null) return List.of();
        return new ArrayList<>(partition.values());
    }

    public List<PropertyRecord> getAll() {
        return store.values().stream()
            .flatMap(m -> m.values().stream())
            .collect(Collectors.toList());
    }

    public List<PropertyRecord> getSalesRecords() {
        List<PropertyRecord> result = new ArrayList<>();
        result.addAll(getBySource("sales"));
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
}
