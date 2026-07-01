package com.houseprices.api;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.houseprices.ingest.DataIngestionService;
import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.PropertyStore;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * REST endpoints to trigger live API fetches from Rentcast and Zillow.
 *
 * Both endpoints require ?confirm=true to prevent accidental calls (billed per call).
 * A cooldown period (default 6h) prevents repeated calls within a short window.
 *
 * POST /api/fetch/rentcast?confirm=true
 * POST /api/fetch/zillow?confirm=true&maxPages=N
 * GET  /api/fetch/status  → shows last fetch times and cooldown state
 */
@Path("/api/fetch")
@Produces(MediaType.APPLICATION_JSON)
public class FetchResource {

    private static final Logger LOG = Logger.getLogger(FetchResource.class);

    @Inject DataIngestionService ingestion;
    @Inject PropertyStore        store;
    @Inject ObjectMapper         mapper;
    @Inject com.houseprices.watcher.FileWatcherService watcher;

    @ConfigProperty(name = "watcher.zillow.dir",   defaultValue = "data/zillow")
    String zillowWatchDir;

    @ConfigProperty(name = "watcher.rentcast.dir", defaultValue = "data/rentcast")
    String rentcastWatchDir;

    /** Minimum hours between API calls. Set to 0 to disable. */
    @ConfigProperty(name = "fetch.cooldown.hours", defaultValue = "6")
    int cooldownHours;

    private Instant lastZillowFetch   = Instant.EPOCH;
    private Instant lastRentcastFetch = Instant.EPOCH;

    // ── Guards ────────────────────────────────────────────────────────────────

    private Map<String, Object> checkGuards(String source, boolean confirm, Instant lastFetch) {
        if (!confirm) {
            return Map.of("status", "error",
                "message", "Add ?confirm=true to confirm. You are billed per API request.");
        }
        if (cooldownHours > 0) {
            Duration since = Duration.between(lastFetch, Instant.now());
            if (since.toHours() < cooldownHours) {
                long minutesLeft = Duration.ofHours(cooldownHours).minus(since).toMinutes();
                return Map.of("status", "error",
                    "message", String.format(
                        "Cooldown active for %s: %d minutes remaining (cooldown=%dh). " +
                        "Override with fetch.cooldown.hours=0 in application.properties.",
                        source, minutesLeft, cooldownHours));
            }
        }
        return null;
    }

    // ── Endpoints ─────────────────────────────────────────────────────────────

    @POST
    @Path("/rentcast")
    public Map<String, Object> fetchRentcast(
            @QueryParam("confirm") @DefaultValue("false") boolean confirm,
            @QueryParam("limit")   @DefaultValue("0")     int limit) {

        Map<String, Object> guard = checkGuards("rentcast", confirm, lastRentcastFetch);
        if (guard != null) return guard;

        try {
            lastRentcastFetch = Instant.now();
            int effectiveLimit = limit > 0 ? limit : 500;

            // Fetch raw JSON from API
            var rawProperties = ingestion.getRentcastApi().fetchRecentSales(effectiveLimit);

            // Save raw JSON to disk (same pattern as Zillow)
            File saveDir  = new File(rentcastWatchDir);
            saveDir.mkdirs();
            File saveFile = new File(saveDir, "rentcast_latest.json");

            watcher.markInProgress(saveFile.getAbsolutePath());
            try {
                mapper.writerWithDefaultPrettyPrinter().writeValue(saveFile,
                    Map.of("requestMetadata", Map.of("status", "ok", "source", "java-api-fetch",
                                                     "limit", effectiveLimit),
                           "properties", rawProperties));
                LOG.infof("Saved RentCast JSON to %s", saveFile.getAbsolutePath());

                List<PropertyRecord> records = ingestion.ingestRentcastJson(saveFile);
                store.upsert(records);
                LOG.infof("Fetched and stored %d RentCast records", records.size());
                return Map.of("status", "ok", "fetched", records.size(),
                              "savedTo", saveFile.getPath(),
                              "nextAllowedIn", cooldownHours + "h");
            } finally {
                watcher.markDone(saveFile.getAbsolutePath());
            }
        } catch (IllegalStateException e) {
            return Map.of("status", "error", "message", e.getMessage());
        } catch (Exception e) {
            LOG.errorf("RentCast fetch failed: %s", e.getMessage());
            return Map.of("status", "error", "message", e.getMessage());
        }
    }

    @POST
    @Path("/zillow")
    public Map<String, Object> fetchZillow(
            @QueryParam("confirm")  @DefaultValue("false") boolean confirm,
            @QueryParam("maxPages") @DefaultValue("0")     int maxPages) {

        Map<String, Object> guard = checkGuards("zillow", confirm, lastZillowFetch);
        if (guard != null) return guard;

        try {
            lastZillowFetch = Instant.now();

            var rawProperties = maxPages > 0
                ? ingestion.getZillowApi().fetchListings(maxPages)
                : ingestion.getZillowApi().fetchListings();

            File saveDir  = new File(zillowWatchDir);
            saveDir.mkdirs();
            File saveFile = new File(saveDir, "zillow_latest.json");

            watcher.markInProgress(saveFile.getAbsolutePath());
            try {
                mapper.writerWithDefaultPrettyPrinter().writeValue(saveFile,
                    Map.of("requestMetadata", Map.of("status", "ok", "source", "java-api-fetch"),
                           "properties", rawProperties));
                LOG.infof("Saved Zillow JSON to %s", saveFile.getAbsolutePath());

                List<PropertyRecord> records = ingestion.ingestZillowJson(saveFile);
                store.upsert(records);
                LOG.infof("Fetched and stored %d Zillow records", records.size());
                return Map.of("status", "ok", "fetched", records.size(),
                              "savedTo", saveFile.getPath(),
                              "nextAllowedIn", cooldownHours + "h");
            } finally {
                watcher.markDone(saveFile.getAbsolutePath());
            }
        } catch (IllegalStateException e) {
            return Map.of("status", "error", "message", e.getMessage());
        } catch (Exception e) {
            LOG.errorf("Zillow fetch failed: %s", e.getMessage());
            return Map.of("status", "error", "message", e.getMessage());
        }
    }

    /** Shows last fetch times and whether cooldown is active */
    @GET
    @Path("/status")
    public Map<String, Object> fetchStatus() {
        Instant now = Instant.now();
        return Map.of(
            "cooldownHours", cooldownHours,
            "zillow",   fetchInfo(lastZillowFetch, now),
            "rentcast", fetchInfo(lastRentcastFetch, now)
        );
    }

    private Map<String, Object> fetchInfo(Instant last, Instant now) {
        if (last.equals(Instant.EPOCH))
            return Map.of("lastFetch", "never", "cooldownActive", false);
        Duration since = Duration.between(last, now);
        boolean active = cooldownHours > 0 && since.toHours() < cooldownHours;
        return Map.of(
            "lastFetch",        last.toString(),
            "minutesAgo",       since.toMinutes(),
            "cooldownActive",   active,
            "minutesRemaining", active ? Duration.ofHours(cooldownHours).minus(since).toMinutes() : 0
        );
    }

    private void saveToCsv(List<PropertyRecord> records, File file) throws java.io.IOException {
        try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(file))) {
            pw.println("id,address,latitude,longitude,lastSalePrice,squareFootage,lotSize,bedrooms,bathrooms,propertyType,lastSaleDate");
            for (PropertyRecord r : records) {
                pw.printf("%s,%s,%.6f,%.6f,%.0f,%.0f,%.0f,%.0f,%.1f,%s,%s%n",
                    csvEscape(r.id()),
                    csvEscape(r.address()),
                    r.lat(), r.lng(),
                    r.salePrice(), r.sqft(), r.sqftLot(),
                    (double) r.beds(), r.baths(),
                    csvEscape(r.homeType()),
                    r.saleDate() != null ? r.saleDate().toString() : ""
                );
            }
        }
    }

    private String csvEscape(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n"))
            return "\"" + s.replace("\"", "\"\"") + "\"";
        return s;
    }
}
