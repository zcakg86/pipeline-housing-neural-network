package com.houseprices.service;

import com.houseprices.ingest.DataIngestionService;
import com.houseprices.ingest.PropertyRecord;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Defers persisted source loading until a dependent map layer is first requested.
 * Live-source files are parsed, de-duplicated, and model-scored as one batch.
 */
@ApplicationScoped
public class LiveSourceLoader {

    private static final Logger LOG = Logger.getLogger(LiveSourceLoader.class);
    private static final Set<String> SOURCES = Set.of("sales", "rentcast", "zillow");

    @Inject DataIngestionService ingestion;
    @Inject PropertyStore store;

    @ConfigProperty(name = "watcher.rentcast.dir", defaultValue = "data/rentcast")
    String rentcastDir;

    @ConfigProperty(name = "watcher.zillow.dir", defaultValue = "data/zillow")
    String zillowDir;

    @ConfigProperty(name = "data.sales.csv", defaultValue = "../../data/sales_2020_25.csv")
    String salesCsv;

    private final Set<String> loaded = ConcurrentHashMap.newKeySet();
    private final ConcurrentHashMap<String, Object> locks = new ConcurrentHashMap<>();

    /** Load a source once. Concurrent map requests wait for the same batch. */
    public void ensureLoaded(String source) {
        if (!SOURCES.contains(source)) {
            throw new IllegalArgumentException("Unknown source: " + source);
        }
        if (loaded.contains(source)) return;

        synchronized (locks.computeIfAbsent(source, ignored -> new Object())) {
            if (loaded.contains(source)) return;
            long started = System.nanoTime();
            List<File> files = "sales".equals(source) ? List.of(new File(salesCsv))
                : sourceFiles("rentcast".equals(source) ? rentcastDir : zillowDir);
            List<PropertyRecord> records;
            try {
                records = switch (source) {
                    case "sales" -> ingestion.ingestSalesCsv(files.getFirst());
                    case "rentcast" -> ingestion.ingestRentcastJsonFiles(files);
                    case "zillow" -> ingestion.ingestZillowJsonFiles(files);
                    default -> throw new IllegalArgumentException("Unknown source: " + source);
                };
            } catch (IOException exception) {
                throw new IllegalStateException("Could not load " + source + " data", exception);
            }
            store.upsert(records);
            loaded.add(source);
            LOG.infof("Lazy-loaded %s: %d file(s), %d record(s), %.1f ms",
                source, files.size(), records.size(), (System.nanoTime() - started) / 1_000_000.0);
        }
    }

    private List<File> sourceFiles(String path) {
        File directory = new File(path);
        File[] files = directory.listFiles(file -> {
            String name = file.getName().toLowerCase();
            return file.isFile() && name.endsWith(".json") && !name.contains("previous");
        });
        if (files == null) return List.of();
        Arrays.sort(files, java.util.Comparator.comparing(File::getName));
        return List.of(files);
    }
}
