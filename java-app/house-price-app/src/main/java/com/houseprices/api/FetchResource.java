package com.houseprices.api;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.houseprices.ingest.RentcastApiClient;
import com.houseprices.ingest.ZillowApiClient;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * REST endpoints to trigger live API fetches from Rentcast and Zillow.
 *
 * Both endpoints require ?confirm=true to prevent accidental calls (billed per call).
 * RentCast has a default 24h cooldown in addition to its persistent call budget.
 *
 * POST /api/fetch/rentcast?confirm=true
 * POST /api/fetch/zillow?confirm=true&maxPages=N
 * GET  /api/fetch/status  → shows last fetch times and cooldown state
 */
@Path("/api/fetch")
@Produces(MediaType.APPLICATION_JSON)
public class FetchResource {

    private static final Logger LOG = Logger.getLogger(FetchResource.class);

    @Inject RentcastApiClient rentcastApi;
    @Inject ZillowApiClient zillowApi;
    @Inject ObjectMapper         mapper;
    @Inject com.houseprices.watcher.FileWatcherService watcher;

    @ConfigProperty(name = "watcher.zillow.dir",   defaultValue = "data/zillow")
    String zillowWatchDir;

    @ConfigProperty(name = "watcher.rentcast.dir", defaultValue = "data/rentcast")
    String rentcastWatchDir;

    /** Minimum hours between API calls. Set to 0 to disable. */
    @ConfigProperty(name = "fetch.cooldown.hours", defaultValue = "0")
    int cooldownHours;

    @ConfigProperty(name = "fetch.rentcast.cooldown.hours", defaultValue = "24")
    int rentcastCooldownHours;

    private Instant lastZillowFetch   = Instant.EPOCH;
    private Instant lastRentcastFetch = Instant.EPOCH;

    // ── Guards ────────────────────────────────────────────────────────────────

    private Map<String, Object> checkGuards(
            String source, boolean confirm, Instant lastFetch, int sourceCooldownHours) {
        if (!confirm) {
            return Map.of("status", "error",
                "message", "Add ?confirm=true to confirm. You are billed per API request.");
        }
        if (sourceCooldownHours > 0) {
            Duration since = Duration.between(lastFetch, Instant.now());
            if (since.toHours() < sourceCooldownHours) {
                long minutesLeft = Duration.ofHours(sourceCooldownHours).minus(since).toMinutes();
                return Map.of("status", "error",
                    "message", String.format(
                        "Cooldown active for %s: %d minutes remaining (cooldown=%dh). " +
                        "Change the source cooldown setting in application.properties to override.",
                        source, minutesLeft, sourceCooldownHours));
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

        Map<String, Object> guard = checkGuards(
            "rentcast", confirm, lastRentcastFetch, rentcastCooldownHours
        );
        if (guard != null) return guard;

        try {
            lastRentcastFetch = Instant.now();
            int effectiveLimit = limit > 0 ? limit : 500;

            int reservedCall = com.houseprices.cli.RentcastFetcher.reserveExternalCall(
                java.nio.file.Path.of(rentcastWatchDir)
            );
            LOG.infof("Reserved persistent RentCast call %d/45", reservedCall);

            // Fetch raw JSON from API
            var rawProperties = rentcastApi.fetchRecentSales(effectiveLimit);

            // Save raw JSON to disk (same pattern as Zillow)
            File saveDir  = new File(rentcastWatchDir);
            saveDir.mkdirs();
            String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS")
            );
            File saveFile = new File(saveDir, "rentcast_" + timestamp + ".json");

            CompletableFuture<com.houseprices.watcher.FileWatcherService.IngestionResult> completion =
                watcher.expect(saveFile);
            try {
                writeJsonAtomically(saveFile,
                    Map.of("requestMetadata", Map.of("status", "ok", "source", "java-api-fetch",
                                                     "limit", effectiveLimit,
                                                     "fetchedAt", Instant.now().toString(),
                                                     "reservedCallNumber", reservedCall),
                           "properties", rawProperties));
                LOG.infof("Saved RentCast JSON to %s", saveFile.getAbsolutePath());
                var result = completion.get(120, TimeUnit.SECONDS);
                return Map.of("status", "ok", "fetched", rawProperties.size(),
                              "ingested", result.parsedAndChanged(),
                              "savedTo", saveFile.getPath(),
                              "nextAllowedIn", rentcastCooldownHours + "h");
            } catch (Exception exception) {
                watcher.cancelExpected(saveFile, completion);
                throw exception;
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

        Map<String, Object> guard = checkGuards(
            "zillow", confirm, lastZillowFetch, cooldownHours
        );
        if (guard != null) return guard;

        try {
            lastZillowFetch = Instant.now();

            var rawProperties = maxPages > 0
                ? zillowApi.fetchListings(maxPages)
                : zillowApi.fetchListings();

            File saveDir = new File(zillowWatchDir);
            saveDir.mkdirs();
            String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS")
            );
            File saveFile = new File(saveDir, "zillow_" + timestamp + ".json");

            CompletableFuture<com.houseprices.watcher.FileWatcherService.IngestionResult> completion =
                watcher.expect(saveFile);
            try {
                writeJsonAtomically(saveFile,
                    Map.of("requestMetadata", Map.of(
                               "status", "ok", "source", "java-api-fetch",
                               "fetchedAt", Instant.now().toString()),
                           "properties", rawProperties));
                LOG.infof("Saved Zillow JSON to %s", saveFile.getAbsolutePath());
                var result = completion.get(120, TimeUnit.SECONDS);
                return Map.of("status", "ok", "fetched", rawProperties.size(),
                              "ingested", result.parsedAndChanged(),
                              "savedTo", saveFile.getPath(),
                              "nextAllowedIn", cooldownHours + "h");
            } catch (Exception exception) {
                watcher.cancelExpected(saveFile, completion);
                throw exception;
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
            "zillow",   fetchInfo(lastZillowFetch, now, cooldownHours),
            "rentcast", fetchInfo(lastRentcastFetch, now, rentcastCooldownHours)
        );
    }

    private Map<String, Object> fetchInfo(Instant last, Instant now, int sourceCooldownHours) {
        if (last.equals(Instant.EPOCH))
            return Map.of(
                "lastFetch", "never", "cooldownActive", false,
                "cooldownHours", sourceCooldownHours
            );
        Duration since = Duration.between(last, now);
        boolean active = sourceCooldownHours > 0 && since.toHours() < sourceCooldownHours;
        return Map.of(
            "lastFetch",        last.toString(),
            "minutesAgo",       since.toMinutes(),
            "cooldownHours",    sourceCooldownHours,
            "cooldownActive",   active,
            "minutesRemaining", active
                ? Duration.ofHours(sourceCooldownHours).minus(since).toMinutes() : 0
        );
    }

    private void writeJsonAtomically(File destination, Object value) throws IOException {
        java.nio.file.Path destinationPath = destination.toPath();
        java.nio.file.Path pendingDir = destinationPath.getParent().resolve(".pending");
        Files.createDirectories(pendingDir);
        java.nio.file.Path temporary = pendingDir.resolve(
            destination.getName() + "." + UUID.randomUUID() + ".tmp"
        );
        mapper.writerWithDefaultPrettyPrinter().writeValue(temporary.toFile(), value);
        try {
            Files.move(
                temporary,
                destinationPath,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            );
        } catch (AtomicMoveNotSupportedException exc) {
            Files.move(temporary, destinationPath, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
