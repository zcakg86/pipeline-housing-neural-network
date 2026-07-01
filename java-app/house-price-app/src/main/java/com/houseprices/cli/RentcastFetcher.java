package com.houseprices.cli;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.nio.file.*;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Standalone CLI tool to fetch Rentcast data without the app running.
 *
 * All API parameters (lat, lng, radius, months, limit, watchDir) are read from
 * src/main/resources/application.properties — the same source used by RentcastApiClient.
 * API key is read from .env.
 *
 * Usage: make rentcast-fetch [LIMIT=500]
 */
public class RentcastFetcher {

    private static final String BASE_URL     = "https://api.rentcast.io/v1/properties";
    private static final String COUNTER_FILE = ".rentcast_call_count";
    private static final int    MAX_CALLS    = 45;

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        int limitOverride = args.length > 0 ? Integer.parseInt(args[0]) : 0;

        // ── Load application.properties (single source of truth for params) ──
        Properties app = loadAppProperties();
        double lat     = Double.parseDouble(app.getProperty("rentcast.lat",    "47.60"));
        double lng     = Double.parseDouble(app.getProperty("rentcast.lng",    "-122.33"));
        int    radius  = Integer.parseInt(app.getProperty("rentcast.radius",   "10"));
        int    months  = Integer.parseInt(app.getProperty("rentcast.months",   "6"));
        int    limit   = limitOverride > 0 ? limitOverride
                       : Integer.parseInt(app.getProperty("rentcast.limit",    "500"));
        String watchDir = app.getProperty("watcher.rentcast.dir", "data/rentcast");

        // ── API key from .env ─────────────────────────────────────────────────
        String apiKey = loadEnvKey("RENTCAST_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("❌ RENTCAST_API_KEY not found in .env");
            System.exit(1);
        }

        // ── Call counter ──────────────────────────────────────────────────────
        Path counterPath = Path.of(COUNTER_FILE);
        int callsUsed = Files.exists(counterPath)
            ? Integer.parseInt(Files.readString(counterPath).trim()) : 0;
        int callsRemaining = MAX_CALLS - callsUsed;

        if (callsRemaining <= 0) {
            System.err.printf("❌ Call limit reached (%d/%d). Edit %s to reset.%n",
                callsUsed, MAX_CALLS, COUNTER_FILE);
            System.exit(1);
        }

        int days = months * 30;
        System.out.printf("Config  : lat=%.2f lng=%.2f radius=%d miles months=%d limit=%d%n",
            lat, lng, radius, months, limit);
        System.out.printf("Budget  : %d/%d calls used, %d remaining%n",
            callsUsed, MAX_CALLS, callsRemaining);

        // ── Paginate ──────────────────────────────────────────────────────────
        ObjectMapper mapper = new ObjectMapper();
        HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();

        List<Object> allProperties = new ArrayList<>();
        int offset     = 0;
        int totalCount = Integer.MAX_VALUE;
        int callsMade  = 0;

        while (offset < totalCount && callsMade < callsRemaining) {
            String url = BASE_URL
                + "?latitude=" + lat
                + "&longitude=" + lng
                + "&radius=" + radius
                + "&limit=" + limit
                + "&offset=" + offset
                + "&includeTotalCount=true"
                + "&saleDateRange=0:" + days;
            //+"&propertyType=Townhouse%7CSingle%20Family";

            System.out.printf("  Call %d/%d — offset=%d, fetching up to %d...%n",
                callsUsed + callsMade + 1, MAX_CALLS, offset, limit);

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("X-Api-Key", apiKey)
                .header("accept", "application/json")
                .GET()
                .build();

            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            callsMade++;

            if (response.statusCode() != 200) {
                System.err.printf("❌ API error %d: %s%n", response.statusCode(),
                    response.body().substring(0, Math.min(300, response.body().length())));
                break;
            }

            Object parsed = mapper.readValue(response.body(), Object.class);
            List<Object> page;

            if (parsed instanceof List) {
                page = (List<Object>) parsed;
                totalCount = offset + page.size();
            } else if (parsed instanceof Map) {
                Map<String, Object> root = (Map<String, Object>) parsed;
                Object tc = root.get("totalCount");
                if (tc instanceof Number && totalCount == Integer.MAX_VALUE) {
                    totalCount = ((Number) tc).intValue();
                    System.out.printf("  Total available: %d%n", totalCount);
                }
                Object props = root.getOrDefault("properties", root.get("results"));
                page = props instanceof List ? (List<Object>) props : List.of();
            } else {
                System.err.println("❌ Unexpected response format"); break;
            }

            if (page.isEmpty()) { System.out.println("  No more records."); break; }

            allProperties.addAll(page);
            offset += page.size();
            System.out.printf("  ✓ %d records (running total: %d)%n", page.size(), allProperties.size());
        }

        if (allProperties.isEmpty()) { System.err.println("❌ No properties fetched."); System.exit(1); }

        // ── Save ──────────────────────────────────────────────────────────────
        Path outDir = Path.of(watchDir);
        Files.createDirectories(outDir);
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        Path outFile = outDir.resolve("rentcast_" + ts + ".json");

        Map<String, Object> output = new LinkedHashMap<>();
        output.put("requestMetadata", Map.of(
            "source", "RentcastFetcher-cli", "limitPerCall", limit,
            "callsMade", callsMade, "totalFetched", allProperties.size(),
            "totalAvailable", totalCount == Integer.MAX_VALUE ? "unknown" : totalCount,
            "timestamp", LocalDateTime.now().toString()
        ));
        output.put("properties", allProperties);
        mapper.writerWithDefaultPrettyPrinter().writeValue(outFile.toFile(), output);

        int newTotal = callsUsed + callsMade;
        Files.writeString(counterPath, String.valueOf(newTotal));

        System.out.printf("%n✓ Saved %d records → %s%n", allProperties.size(), outFile);
        System.out.printf("  Calls used: %d/%d (%d remaining)%n", newTotal, MAX_CALLS, MAX_CALLS - newTotal);
        if (totalCount != Integer.MAX_VALUE && allProperties.size() < totalCount)
            System.out.printf("  ⚠ %d records remaining (next offset: %d) — run again to continue%n",
                totalCount - allProperties.size(), offset);
    }

    /** Reads src/main/resources/application.properties */
    private static Properties loadAppProperties() throws IOException {
        Properties p = new Properties();
        Path path = Path.of("src/main/resources/application.properties");
        if (Files.exists(path)) {
            try (InputStream is = Files.newInputStream(path)) { p.load(is); }
        } else {
            System.err.println("⚠ application.properties not found, using defaults");
        }
        return p;
    }

    private static String loadEnvKey(String key) throws IOException {
        Path envPath = Path.of(".env");
        if (!Files.exists(envPath)) return System.getenv(key);
        for (String line : Files.readAllLines(envPath)) {
            line = line.trim();
            if (line.startsWith(key + "=")) return line.substring(key.length() + 1).trim();
        }
        return System.getenv(key);
    }
}
