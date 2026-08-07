package com.houseprices.cli;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.houseprices.ingest.RentcastUniqueId;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;

/**
 * Charge-guarded standalone RentCast fetcher.
 *
 * No request is made without --confirm. One invocation makes one call by
 * default; pagination requires an explicit --max-calls value and is capped at
 * five calls per run. Every call is reserved in a persistent counter before
 * the HTTP request begins, and every received response is atomically saved in
 * the {requestMetadata, properties} envelope consumed by DataIngestionService.
 */
public class RentcastFetcher {

    private static final String BASE_URL = "https://api.rentcast.io/v1/properties";
    private static final int MAX_TOTAL_CALLS = 45;
    private static final int MAX_CALLS_PER_RUN = 5;
    private static final int MAX_PAGE_SIZE = 500;
    private static final DateTimeFormatter FILE_TIMESTAMP =
        DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS");

    private record Options(int limit, int maxCalls, int offset, boolean confirmed) {}

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        Options options = parseOptions(args);
        if (!options.confirmed()) {
            System.err.println("No API request made. Re-run with --confirm after reviewing the budget.");
            printUsage();
            System.exit(2);
        }
        if (options.limit() < 1 || options.limit() > MAX_PAGE_SIZE) {
            throw new IllegalArgumentException("--limit must be between 1 and " + MAX_PAGE_SIZE);
        }
        if (options.maxCalls() < 1 || options.maxCalls() > MAX_CALLS_PER_RUN) {
            throw new IllegalArgumentException(
                "--max-calls must be between 1 and " + MAX_CALLS_PER_RUN
            );
        }
        if (options.offset() < 0) {
            throw new IllegalArgumentException("--offset cannot be negative");
        }

        Properties app = loadAppProperties();
        double lat = Double.parseDouble(app.getProperty("rentcast.lat", "47.60"));
        double lng = Double.parseDouble(app.getProperty("rentcast.lng", "-122.33"));
        int radius = Integer.parseInt(app.getProperty("rentcast.radius", "10"));
        int months = Integer.parseInt(app.getProperty("rentcast.months", "4"));
        int requestTimeoutSeconds = Integer.parseInt(
            app.getProperty("rentcast.timeout.seconds", "180")
        );
        String watchDir = app.getProperty("watcher.rentcast.dir", "data/rentcast");

        if (months < 1) {
            throw new IllegalArgumentException("rentcast.months must be at least 1");
        }
        if (requestTimeoutSeconds < 30 || requestTimeoutSeconds > 600) {
            throw new IllegalArgumentException(
                "rentcast.timeout.seconds must be between 30 and 600"
            );
        }

        String apiKey = loadEnvKey("RENTCAST_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("RENTCAST_API_KEY not found in .env or environment");
        }

        Path outDir = Path.of(watchDir);
        Path stateDir = outDir.resolve(".state");
        Files.createDirectories(stateDir);
        Path counterPath = stateDir.resolve("call_count");
        Path lockPath = stateDir.resolve("fetch.lock");

        try (FileChannel lockChannel = FileChannel.open(
                 lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE
             ); FileLock ignored = acquireExclusiveLock(lockChannel)) {

            int callsUsed = loadCurrentUsage(outDir, counterPath);
            int callsRemaining = MAX_TOTAL_CALLS - callsUsed;
            if (callsRemaining <= 0) {
                throw new IllegalStateException(
                    "Persistent RentCast call limit reached (" + callsUsed + "/" +
                    MAX_TOTAL_CALLS + ")"
                );
            }
            if (options.maxCalls() > callsRemaining) {
                throw new IllegalStateException(
                    "Requested " + options.maxCalls() + " calls but only " +
                    callsRemaining + " remain in the persistent budget"
                );
            }

            System.out.printf(
                "Config  : lat=%.2f lng=%.2f radius=%d miles months=%d " +
                "limit=%d offset=%d timeout=%ds%n",
                lat, lng, radius, months, options.limit(), options.offset(),
                requestTimeoutSeconds
            );
            System.out.printf(
                "Budget  : %d/%d used; this run is capped at %d call(s)%n",
                callsUsed, MAX_TOTAL_CALLS, options.maxCalls()
            );
            System.out.println("Stop     : Ctrl-C is safe; calls are counted before sending and pages save immediately.");

            ObjectMapper mapper = new ObjectMapper();
            HttpClient http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();

            String runTimestamp = LocalDateTime.now().format(FILE_TIMESTAMP);
            List<Object> allProperties = new ArrayList<>();
            Set<String> seenIds = new LinkedHashSet<>();
            int offset = options.offset();
            int totalCount = Integer.MAX_VALUE;
            int callsMade = 0;

            while (callsMade < options.maxCalls() && offset < totalCount) {
                int reservedTotal = callsUsed + callsMade + 1;
                writeCounterAtomically(counterPath, reservedTotal);
                callsMade++;

                String url = buildUrl(
                    lat, lng, radius, options.limit(), offset, months * 30
                );
                System.out.printf(
                    "  Reserved call %d/%d — offset=%d, limit=%d%n",
                    reservedTotal, MAX_TOTAL_CALLS, offset, options.limit()
                );

                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofSeconds(requestTimeoutSeconds))
                    .header("X-Api-Key", apiKey)
                    .header("accept", "application/json")
                    .GET()
                    .build();

                // Deliberately no automatic retry: one send equals at most one billable call.
                Path pageFile = outDir.resolve(String.format(
                    "rentcast_%s_call%02d_offset%06d.json",
                    runTimestamp, callsMade, offset
                ));
                HttpResponse<String> response;
                try {
                    response = http.send(request, HttpResponse.BodyHandlers.ofString());
                } catch (IOException | InterruptedException exc) {
                    Map<String, Object> failureMetadata = new LinkedHashMap<>();
                    failureMetadata.put("status", "transport-error");
                    failureMetadata.put("source", "RentcastFetcher-cli");
                    failureMetadata.put("reservedCallNumber", reservedTotal);
                    failureMetadata.put("runCallNumber", callsMade);
                    failureMetadata.put("offset", offset);
                    failureMetadata.put("limit", options.limit());
                    failureMetadata.put("months", months);
                    failureMetadata.put("requestTimeoutSeconds", requestTimeoutSeconds);
                    failureMetadata.put("requestUrl", url);
                    failureMetadata.put("errorType", exc.getClass().getSimpleName());
                    failureMetadata.put("errorMessage", String.valueOf(exc.getMessage()));
                    failureMetadata.put("timestamp", LocalDateTime.now().toString());
                    writeEnvelopeAtomically(
                        mapper, outDir, pageFile, failureMetadata, List.of()
                    );
                    System.err.printf("Saved failed attempt metadata → %s%n", pageFile);
                    if (exc instanceof InterruptedException) {
                        Thread.currentThread().interrupt();
                    }
                    throw exc;
                }

                List<Object> page = List.of();
                String parseError = null;
                if (response.statusCode() == 200) {
                    try {
                        Object parsed = mapper.readValue(response.body(), Object.class);
                        if (parsed instanceof List<?> list) {
                            page = (List<Object>) list;
                        } else if (parsed instanceof Map<?, ?> rawRoot) {
                            Map<String, Object> root = (Map<String, Object>) rawRoot;
                            Object tc = root.get("totalCount");
                            if (tc instanceof Number number) {
                                totalCount = number.intValue();
                            }
                            Object properties = root.getOrDefault("properties", root.get("results"));
                            page = properties instanceof List<?> list
                                ? (List<Object>) list : List.of();
                        } else {
                            parseError = "Unexpected JSON root type";
                        }
                    } catch (Exception exc) {
                        parseError = exc.getClass().getSimpleName() + ": " + exc.getMessage();
                    }
                }

                Map<String, Object> metadata = new LinkedHashMap<>();
                metadata.put("status", response.statusCode() == 200 && parseError == null ? "ok" : "error");
                metadata.put("source", "RentcastFetcher-cli");
                metadata.put("httpStatus", response.statusCode());
                metadata.put("reservedCallNumber", reservedTotal);
                metadata.put("runCallNumber", callsMade);
                metadata.put("offset", offset);
                metadata.put("limit", options.limit());
                metadata.put("requestUrl", url);
                metadata.put("propertiesReturned", page.size());
                metadata.put("totalAvailable", totalCount == Integer.MAX_VALUE ? "unknown" : totalCount);
                metadata.put("timestamp", LocalDateTime.now().toString());
                if (parseError != null) metadata.put("parseError", parseError);
                if (response.statusCode() != 200 || parseError != null) {
                    metadata.put("rawResponseBody", response.body());
                }

                // Add the canonical numeric ID before either the page or the
                // aggregate is written. This matches the in-app API client.
                for (Object property : page) {
                    if (property instanceof Map<?, ?> rawProperty) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> propertyMap = (Map<String, Object>) rawProperty;
                        RentcastUniqueId.addTo(propertyMap);
                    }
                }

                writeEnvelopeAtomically(mapper, outDir, pageFile, metadata, page);
                System.out.printf(
                    "  Saved response immediately: HTTP %d, %d properties → %s%n",
                    response.statusCode(), page.size(), pageFile
                );

                if (response.statusCode() != 200 || parseError != null || page.isEmpty()) {
                    break;
                }

                int newIds = 0;
                for (Object property : page) {
                    if (property instanceof Map<?, ?> record) {
                        String id = RentcastUniqueId.storeKey(record.get(RentcastUniqueId.JSON_FIELD));
                        if (!id.isBlank() && seenIds.add(id)) newIds++;
                    }
                }
                if (!seenIds.isEmpty() && newIds == 0) {
                    System.err.println("Stopped: API returned a page containing no new property IDs.");
                    break;
                }

                allProperties.addAll(page);
                offset += page.size();
                if (offset >= totalCount || page.size() < options.limit()) break;
            }

            if (!allProperties.isEmpty()) {
                Map<String, Object> aggregateMetadata = new LinkedHashMap<>();
                aggregateMetadata.put("status", "ok");
                aggregateMetadata.put("source", "RentcastFetcher-cli-aggregate");
                aggregateMetadata.put("callsMade", callsMade);
                aggregateMetadata.put("startOffset", options.offset());
                aggregateMetadata.put("nextOffset", offset);
                aggregateMetadata.put("totalFetched", allProperties.size());
                aggregateMetadata.put(
                    "totalAvailable", totalCount == Integer.MAX_VALUE ? "unknown" : totalCount
                );
                aggregateMetadata.put("timestamp", LocalDateTime.now().toString());
                Path aggregateFile = outDir.resolve("rentcast_" + runTimestamp + "_aggregate.json");
                writeEnvelopeAtomically(
                    mapper, outDir, aggregateFile, aggregateMetadata, allProperties
                );
                System.out.printf("%nSaved aggregate: %d properties → %s%n", allProperties.size(), aggregateFile);
            }

            int newTotal = callsUsed + callsMade;
            System.out.printf(
                "Calls reserved: %d/%d (%d remaining)%n",
                newTotal, MAX_TOTAL_CALLS, MAX_TOTAL_CALLS - newTotal
            );
            if (callsMade == options.maxCalls() && offset < totalCount) {
                System.out.printf(
                    "More records may exist. Review saved pages, then explicitly continue with --offset=%d.%n",
                    offset
                );
            }
        }
    }

    private static String buildUrl(
        double lat, double lng, int radius, int limit, int offset, int days
    ) {
        return BASE_URL
            + "?latitude=" + lat
            + "&longitude=" + lng
            + "&radius=" + radius
            + "&limit=" + limit
            + "&offset=" + offset
            + "&includeTotalCount=true"
            + "&saleDateRange=0:" + days
            + "&propertyType=Townhouse%7CSingle%20Family";
    }

    private static Options parseOptions(String[] args) {
        int limit = 0;
        int maxCalls = 1;
        int offset = 0;
        boolean confirmed = false;
        for (String arg : args) {
            if (arg.equals("--confirm")) confirmed = true;
            else if (arg.startsWith("--limit=")) limit = parseIntOption(arg, "--limit=");
            else if (arg.startsWith("--max-calls=")) maxCalls = parseIntOption(arg, "--max-calls=");
            else if (arg.startsWith("--offset=")) offset = parseIntOption(arg, "--offset=");
            else if (arg.matches("\\d+") && limit == 0) limit = Integer.parseInt(arg);
            else throw new IllegalArgumentException("Unknown argument: " + arg);
        }
        if (limit == 0) {
            try {
                limit = Integer.parseInt(loadAppProperties().getProperty("rentcast.limit", "500"));
            } catch (IOException exc) {
                throw new IllegalStateException("Could not load default RentCast limit", exc);
            }
        }
        return new Options(limit, maxCalls, offset, confirmed);
    }

    private static int parseIntOption(String argument, String prefix) {
        return Integer.parseInt(argument.substring(prefix.length()));
    }

    private static void printUsage() {
        System.err.println(
            "Usage: RentcastFetcher --confirm [--limit=500] [--max-calls=1] [--offset=0]"
        );
    }

    private static FileLock acquireExclusiveLock(FileChannel channel) throws IOException {
        try {
            FileLock lock = channel.tryLock();
            if (lock == null) throw new IOException("Another RentcastFetcher is already running");
            return lock;
        } catch (OverlappingFileLockException exc) {
            throw new IOException("Another RentcastFetcher is already running", exc);
        }
    }

    private static int readCounterFailClosed(Path counterPath) throws IOException {
        if (!Files.exists(counterPath)) return 0;
        String raw = Files.readString(counterPath).trim();
        try {
            int value = Integer.parseInt(raw);
            if (value < 0 || value > MAX_TOTAL_CALLS) throw new NumberFormatException();
            return value;
        } catch (NumberFormatException exc) {
            throw new IOException(
                "Invalid call counter at " + counterPath + "; refusing to make an API request",
                exc
            );
        }
    }

    private static int loadCurrentUsage(Path outDir, Path counterPath) throws IOException {
        boolean hasPersistentCounter = Files.exists(counterPath);
        int callsUsed = readCounterFailClosed(counterPath);
        // Preserve the budget from the original CLI counter location so
        // upgrading this safety logic cannot accidentally reset usage.
        int legacyCallsUsed = readCounterFailClosed(Path.of(".rentcast_call_count"));
        if (legacyCallsUsed > callsUsed) callsUsed = legacyCallsUsed;
        if (!hasPersistentCounter) {
            callsUsed = Math.max(callsUsed, inferCallsFromSavedResponses(outDir));
        }
        if (!hasPersistentCounter || callsUsed != readCounterFailClosed(counterPath)) {
            writeCounterAtomically(counterPath, callsUsed);
        }
        return callsUsed;
    }

    /**
     * Reserve one billable call for the in-app REST fetch path. It shares the
     * same fail-closed counter and exclusive lock as this CLI.
     */
    public static int reserveExternalCall(Path outDir) throws IOException {
        Path stateDir = outDir.resolve(".state");
        Files.createDirectories(stateDir);
        Path counterPath = stateDir.resolve("call_count");
        Path lockPath = stateDir.resolve("fetch.lock");
        try (FileChannel channel = FileChannel.open(
                 lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE
             ); FileLock ignored = acquireExclusiveLock(channel)) {
            int callsUsed = loadCurrentUsage(outDir, counterPath);
            if (callsUsed >= MAX_TOTAL_CALLS) {
                throw new IOException(
                    "Persistent RentCast call limit reached (" + callsUsed + "/" +
                    MAX_TOTAL_CALLS + ")"
                );
            }
            int reserved = callsUsed + 1;
            writeCounterAtomically(counterPath, reserved);
            return reserved;
        }
    }

    @SuppressWarnings("unchecked")
    private static int inferCallsFromSavedResponses(Path outDir) throws IOException {
        if (!Files.isDirectory(outDir)) return 0;
        ObjectMapper mapper = new ObjectMapper();
        int inferred = 0;
        try (var paths = Files.list(outDir)) {
            for (Path path : paths.filter(
                candidate -> candidate.getFileName().toString().toLowerCase().endsWith(".json")
            ).toList()) {
                try {
                    Map<String, Object> root = mapper.readValue(path.toFile(), Map.class);
                    Object rawMetadata = root.get("requestMetadata");
                    if (!(rawMetadata instanceof Map<?, ?> metadata)) continue;
                    Object rawSource = metadata.get("source");
                    String source = rawSource == null ? "" : String.valueOf(rawSource);
                    if (source.equals("RentcastFetcher-cli-aggregate")) continue;
                    Object callsMade = metadata.get("callsMade");
                    if (source.equals("RentcastFetcher-cli") && callsMade instanceof Number number) {
                        inferred += number.intValue();
                    } else if (source.equals("RentcastFetcher-cli") || source.equals("java-api-fetch")) {
                        inferred++;
                    }
                } catch (Exception ignored) {
                    // An unreadable response could still represent a billed
                    // call, so count it conservatively rather than resetting.
                    inferred++;
                }
            }
        }
        return Math.min(inferred, MAX_TOTAL_CALLS);
    }

    private static void writeCounterAtomically(Path counterPath, int value) throws IOException {
        Path temporary = counterPath.resolveSibling(
            "." + counterPath.getFileName() + "." + UUID.randomUUID() + ".tmp"
        );
        Files.writeString(
            temporary,
            String.valueOf(value),
            StandardOpenOption.CREATE_NEW,
            StandardOpenOption.WRITE
        );
        moveAtomically(temporary, counterPath);
    }

    private static void writeEnvelopeAtomically(
        ObjectMapper mapper,
        Path outDir,
        Path destination,
        Map<String, Object> metadata,
        List<Object> properties
    ) throws IOException {
        Path pendingDir = outDir.resolve(".pending");
        Files.createDirectories(pendingDir);
        Path temporary = pendingDir.resolve(destination.getFileName() + "." + UUID.randomUUID() + ".tmp");
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("requestMetadata", metadata);
        envelope.put("properties", properties);
        mapper.writerWithDefaultPrettyPrinter().writeValue(temporary.toFile(), envelope);
        moveAtomically(temporary, destination);
    }

    private static void moveAtomically(Path source, Path destination) throws IOException {
        try {
            Files.move(
                source,
                destination,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            );
        } catch (AtomicMoveNotSupportedException exc) {
            Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Reads src/main/resources/application.properties. */
    private static Properties loadAppProperties() throws IOException {
        Properties properties = new Properties();
        Path path = Path.of("src/main/resources/application.properties");
        if (Files.exists(path)) {
            try (InputStream input = Files.newInputStream(path)) {
                properties.load(input);
            }
        } else {
            System.err.println("application.properties not found; using safe defaults");
        }
        return properties;
    }

    private static String loadEnvKey(String key) throws IOException {
        Path envPath = Path.of(".env");
        if (!Files.exists(envPath)) return System.getenv(key);
        for (String line : Files.readAllLines(envPath)) {
            line = line.trim();
            if (line.startsWith(key + "=")) {
                return line.substring(key.length() + 1).trim();
            }
        }
        return System.getenv(key);
    }
}
