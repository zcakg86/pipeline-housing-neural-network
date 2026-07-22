package com.houseprices.watcher;

import com.houseprices.ingest.DataIngestionService;
import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.PropertyStore;
import io.quarkus.runtime.StartupEvent;
import jakarta.annotation.PreDestroy;
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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Watches configured directories for new canonical JSON files.
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

    private final ConcurrentHashMap<String, CompletableFuture<IngestionResult>> expectedFiles
        = new ConcurrentHashMap<>();
    private final Set<String> inProgressFiles = ConcurrentHashMap.newKeySet();
    private final Set<String> processedFiles = ConcurrentHashMap.newKeySet();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile WatchService watchService;

    public record IngestionResult(int parsedAndChanged, String file) {}

    void onStart(@Observes StartupEvent ev) {
        if (!running.compareAndSet(false, true)) {
            LOG.warn("File watcher start requested while it is already running");
            return;
        }
        ensureDir(rentcastDir);
        ensureDir(zillowDir);
        executor.submit(this::watchLoop);
        LOG.infof("File watcher started. Watching: %s, %s", rentcastDir, zillowDir);
    }

    private void watchLoop() {
        try {
            WatchService watcher = FileSystems.getDefault().newWatchService();
            watchService = watcher;
            if (!running.get()) {
                watcher.close();
                return;
            }
            Path rentcast = Path.of(rentcastDir);
            Path zillow   = Path.of(zillowDir);

            rentcast.register(watcher, StandardWatchEventKinds.ENTRY_CREATE);
            zillow.register(watcher,   StandardWatchEventKinds.ENTRY_CREATE);

            while (running.get() && !Thread.currentThread().isInterrupted()) {
                WatchKey key = watcher.take();  // blocks
                Path dir = (Path) key.watchable();

                for (WatchEvent<?> event : key.pollEvents()) {
                    if (event.kind() == StandardWatchEventKinds.OVERFLOW) continue;
                    Path filename = (Path) event.context();
                    File file = dir.resolve(filename).toFile();
                    String lowerName = file.getName().toLowerCase();
                    if (!file.isFile() || !lowerName.endsWith(".json")
                            || lowerName.contains("previous")) {
                        LOG.debugf("Ignoring non-JSON watcher entry: %s", file.getPath());
                        continue;
                    }
                    processNewFile(file, dir.equals(rentcast) ? "rentcast" : "zillow");
                }
                if (!key.reset()) {
                    LOG.warnf("Stopped watching unavailable directory: %s", dir);
                    break;
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (ClosedWatchServiceException e) {
            if (running.get()) LOG.warn("File watcher closed unexpectedly");
        } catch (IOException e) {
            if (running.get()) LOG.errorf("File watcher error: %s", e.getMessage());
        } finally {
            watchService = null;
            running.set(false);
        }
    }

    private void processNewFile(File file, String source) {
        String name = file.getName().toLowerCase();
        String key = normalizedKey(file);
        if (processedFiles.contains(key) || !inProgressFiles.add(key)) {
            LOG.debugf("Ignoring duplicate watcher event: %s", file.getPath());
            return;
        }
        CompletableFuture<IngestionResult> completion = expectedFiles.get(key);

        LOG.infof("New file detected [%s]: %s", source, file.getName());
        try {
            // Wait for file to finish writing (size stabilises)
            waitForFile(file);

            List<PropertyRecord> records;
            if (source.equals("rentcast") && name.endsWith(".json")) {
                records = ingestion.ingestRentcastJson(file);
            } else if (source.equals("zillow") && name.endsWith(".json")) {
                records = ingestion.ingestZillowJson(file);
            } else {
                LOG.infof("Skipping unsupported file type: %s", file.getName());
                if (completion != null) completion.completeExceptionally(
                    new IllegalArgumentException("Unsupported file: " + file.getName())
                );
                return;
            }
            store.upsert(records);
            processedFiles.add(key);
            LOG.infof("Processed %d records from %s", records.size(), file.getName());
            if (completion != null) {
                completion.complete(new IngestionResult(records.size(), file.getPath()));
            }
        } catch (Exception e) {
            LOG.errorf("Failed to process file %s: %s", file.getName(), e.getMessage());
            if (completion != null) completion.completeExceptionally(e);
        } finally {
            inProgressFiles.remove(key);
            if (completion != null) expectedFiles.remove(key, completion);
        }
    }

    @PreDestroy
    void stop() {
        boolean wasRunning = running.getAndSet(false);

        WatchService watcher = watchService;
        if (watcher != null) {
            try {
                watcher.close();
            } catch (IOException e) {
                LOG.debugf("Error closing file watcher: %s", e.getMessage());
            }
        }
        executor.shutdownNow();
        IllegalStateException stopped = new IllegalStateException(
            "File watcher stopped during application reload"
        );
        expectedFiles.forEach((key, future) -> future.completeExceptionally(stopped));
        expectedFiles.clear();
        inProgressFiles.clear();
        processedFiles.clear();
        if (wasRunning) LOG.info("File watcher stopped");
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

    /** Register an app-generated file before its atomic move into the watched directory. */
    public CompletableFuture<IngestionResult> expect(File file) {
        CompletableFuture<IngestionResult> future = new CompletableFuture<>();
        CompletableFuture<IngestionResult> previous = expectedFiles.putIfAbsent(
            normalizedKey(file), future
        );
        if (previous != null) {
            throw new IllegalStateException("File is already queued: " + file.getPath());
        }
        return future;
    }

    public void cancelExpected(File file, CompletableFuture<IngestionResult> future) {
        expectedFiles.remove(normalizedKey(file), future);
        future.cancel(false);
    }

    private String normalizedKey(File file) {
        return file.toPath().toAbsolutePath().normalize().toString();
    }

    private void ensureDir(String path) {
        File dir = new File(path);
        if (!dir.exists()) dir.mkdirs();
    }
}
