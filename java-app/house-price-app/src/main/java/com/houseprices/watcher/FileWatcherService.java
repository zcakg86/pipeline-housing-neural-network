package com.houseprices.watcher;

import com.houseprices.ingest.DataIngestionService;
import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.PropertyStore;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Watches configured directories for new CSV files.
 * - data/rentcast/   → ingested as Rentcast sales data
 * - data/zillow/     → ingested as Zillow listings (JSON)
 *
 * New files are processed immediately on arrival.
 */
@ApplicationScoped
public class FileWatcherService {

    private static final Logger LOG = Logger.getLogger(FileWatcherService.class);

    @Inject DataIngestionService ingestion;
    @Inject PropertyStore        store;

    @ConfigProperty(name = "watcher.rentcast.dir", defaultValue = "data/rentcast")
    String rentcastDir;

    @ConfigProperty(name = "watcher.zillow.dir", defaultValue = "data/zillow")
    String zillowDir;

    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "file-watcher");
        t.setDaemon(true);
        return t;
    });

    // Tracks files currently being processed by FetchResource to avoid double-ingest
    private final Set<String> inProgress = ConcurrentHashMap.newKeySet();

    void onStart(@Observes StartupEvent ev) {
        ensureDir(rentcastDir);
        ensureDir(zillowDir);
        executor.submit(this::watchLoop);
        LOG.infof("File watcher started. Watching: %s, %s", rentcastDir, zillowDir);
    }

    private void watchLoop() {
        try {
            WatchService watcher = FileSystems.getDefault().newWatchService();
            Path rentcast = Path.of(rentcastDir);
            Path zillow   = Path.of(zillowDir);

            rentcast.register(watcher, StandardWatchEventKinds.ENTRY_CREATE);
            zillow.register(watcher,   StandardWatchEventKinds.ENTRY_CREATE);

            while (!Thread.currentThread().isInterrupted()) {
                WatchKey key = watcher.take();  // blocks
                Path dir = (Path) key.watchable();

                for (WatchEvent<?> event : key.pollEvents()) {
                    if (event.kind() == StandardWatchEventKinds.OVERFLOW) continue;
                    Path filename = (Path) event.context();
                    File file = dir.resolve(filename).toFile();
                    processNewFile(file, dir.equals(rentcast) ? "rentcast" : "zillow");
                }
                key.reset();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (IOException e) {
            LOG.errorf("File watcher error: %s", e.getMessage());
        }
    }

    private void processNewFile(File file, String source) {
        String name = file.getName().toLowerCase();
        String key  = file.getAbsolutePath();

        // Skip if FetchResource already ingested this file
        if (!inProgress.add(key)) {
            LOG.debugf("Skipping already-processed file: %s", file.getName());
            return;
        }

        LOG.infof("New file detected [%s]: %s", source, file.getName());
        try {
            // Wait for file to finish writing (size stabilises)
            waitForFile(file);

            List<PropertyRecord> records;
            if (source.equals("rentcast") && name.endsWith(".csv")) {
                records = ingestion.ingestRentcastCsv(file);
            } else if (source.equals("rentcast") && name.endsWith(".json")) {
                records = ingestion.ingestRentcastJson(file);
            } else if (source.equals("zillow") && name.endsWith(".json")) {
                records = ingestion.ingestZillowJson(file);
            } else if (source.equals("zillow") && name.endsWith(".csv")) {
                records = ingestion.ingestSalesCsv(file);
            } else {
                LOG.infof("Skipping unsupported file type: %s", file.getName());
                return;
            }
            store.upsert(records);
            LOG.infof("Processed %d records from %s", records.size(), file.getName());
        } catch (Exception e) {
            LOG.errorf("Failed to process file %s: %s", file.getName(), e.getMessage());
        } finally {
            inProgress.remove(key);
        }
    }

    private void waitForFile(File file) throws InterruptedException {
        long previousSize = -1;
        for (int i = 0; i < 10; i++) {
            long currentSize = file.length();
            if (currentSize > 0 && currentSize == previousSize) return;
            previousSize = currentSize;
            Thread.sleep(200);
        }
    }

    /** Called by FetchResource before writing a file, so the watcher skips it */
    public void markInProgress(String absolutePath) { inProgress.add(absolutePath); }
    public void markDone(String absolutePath)        { inProgress.remove(absolutePath); }

    private void ensureDir(String path) {
        File dir = new File(path);
        if (!dir.exists()) dir.mkdirs();
    }
}
