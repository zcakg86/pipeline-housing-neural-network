package com.houseprices.api;

import com.houseprices.ingest.DataIngestionService;
import com.houseprices.service.PropertyStore;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.File;
import java.util.Arrays;

/**
 * Loads initial data from configured CSV/JSON files on startup.
 */
@ApplicationScoped
public class StartupLoader {

    private static final Logger LOG = Logger.getLogger(StartupLoader.class);

    @Inject DataIngestionService ingestion;
    @Inject PropertyStore        store;

    @ConfigProperty(name = "data.sales.csv",  defaultValue = "../../data/sales_2020_25.csv")
    String salesCsvPath;

    @ConfigProperty(name = "watcher.rentcast.dir", defaultValue = "data/rentcast")
    String rentcastWatchDir;

    @ConfigProperty(name = "watcher.zillow.dir", defaultValue = "data/zillow")
    String zillowWatchDir;

    void onStart(@Observes StartupEvent ev) {
        LOG.info("Loading initial data...");

        // Load historical baseline sales data
        loadFile(salesCsvPath, "sales");

        // Load any previously fetched files saved in the watcher directories
        loadWatchDir(rentcastWatchDir, "rentcast");
        loadWatchDir(zillowWatchDir,   "zillow");

        LOG.infof("Startup load complete. Total records: %d | %s",
            store.totalCount(), store.countsBySource());
    }

    private void loadWatchDir(String dirPath, String type) {
        File dir = new File(dirPath);
        if (!dir.exists() || !dir.isDirectory()) return;
        File[] files = dir.listFiles();
        if (files == null) return;
        Arrays.sort(files, java.util.Comparator.comparing(File::getName));
        for (File file : files) {
            String name = file.getName().toLowerCase();
            if (!file.isFile() || !name.endsWith(".json") || name.contains("previous")) {
                LOG.debugf("Skipping non-JSON startup entry: %s", file.getPath());
                continue;
            }
            loadFile(file.getPath(), type);
        }
    }

    private void loadFile(String path, String type) {
        File file = new File(path);
        if (!file.isFile()) {
            LOG.warnf("Startup data file not found, skipping: %s", path);
            return;
        }
        String name = file.getName().toLowerCase();
        try {
            var records = switch (type) {
                case "sales"    -> ingestion.ingestSalesCsv(file);
                case "rentcast" -> ingestion.ingestRentcastJson(file);
                case "zillow"   -> ingestion.ingestZillowJson(file);
                default         -> throw new IllegalArgumentException("Unknown type: " + type);
            };
            store.upsert(records);
            LOG.infof("Loaded %d %s records from %s", records.size(), type, path);
        } catch (Exception e) {
            LOG.errorf("Failed to load %s from %s: %s", type, path, e.getMessage());
        }
    }
}
