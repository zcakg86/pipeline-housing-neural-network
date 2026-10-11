package com.houseprices.ingest;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Calls the HasData Redfin listing scraper and retains every complete page response. */
@ApplicationScoped
public class RedfinApiClient {

    private static final Logger LOG = Logger.getLogger(RedfinApiClient.class);
    private static final Pattern PAGE_IN_URL = Pattern.compile("(?:/page-|[?&]page=)(\\d+)");

    @Inject ObjectMapper mapper;

    @ConfigProperty(name = "redfin.api.url", defaultValue = "https://api.hasdata.com/scrape/redfin/listing")
    String apiUrl;

    // Redfin and Zillow are both HasData APIs and intentionally share this key.
    @ConfigProperty(name = "zillow.api.key", defaultValue = "")
    String apiKey;

    @ConfigProperty(name = "redfin.keywords", defaultValue = "98144")
    String keywords;

    @ConfigProperty(name = "redfin.type", defaultValue = "forSale")
    String listingType;

    @ConfigProperty(name = "redfin.sort", defaultValue = "newest")
    String sort;

    @ConfigProperty(name = "redfin.home.types", defaultValue = "condo,townhouse,house")
    String homeTypes;

    @ConfigProperty(name = "redfin.time.on.redfin")
    Optional<String> timeOnRedfin;

    @ConfigProperty(name = "redfin.fetch.all.pages", defaultValue = "false")
    boolean fetchAllPages;

    @ConfigProperty(name = "redfin.max.pages", defaultValue = "20")
    int maxPages;

    @ConfigProperty(name = "redfin.delay.ms", defaultValue = "2000")
    long delayMs;

    private final HttpClient http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .build();

    /** Fetch using the configured all-pages option. */
    public FetchResult fetchListings() throws Exception {
        return fetchListings(fetchAllPages, (keyword, page, response) -> {});
    }

    /** Fetch using the configured all-pages option with page-by-page handling. */
    public FetchResult fetchListings(PageHandler pageHandler) throws Exception {
        return fetchListings(fetchAllPages, pageHandler);
    }

    /** Fetch one page, or follow response pagination until exhausted or the safety cap is reached. */
    public FetchResult fetchListings(boolean allPages) throws Exception {
        return fetchListings(allPages, (keyword, page, response) -> {});
    }

    /**
     * Fetch listings and durably handle each successful response before issuing
     * the next billed request.
     */
    @SuppressWarnings("unchecked")
    public FetchResult fetchListings(boolean allPages, PageHandler pageHandler) throws Exception {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("zillow.api.key is not configured");
        }
        if (maxPages < 1) throw new IllegalStateException("redfin.max.pages must be positive");

        List<Map<String, Object>> responses = new ArrayList<>();
        List<Map<String, Object>> properties = new ArrayList<>();
        List<String> searchKeywords = configuredKeywords();
        int requestCount = 0;
        FetchFailure failure = null;
        boolean stop = false;
        for (String searchKeyword : searchKeywords) {
            Set<Integer> requestedPages = new LinkedHashSet<>();
            int page = 1;
            int pagesForKeyword = 0;
            while (pagesForKeyword < maxPages && requestedPages.add(page)) {
                if (requestCount > 0 && delayMs > 0) Thread.sleep(delayMs);
                String url = buildUrl(searchKeyword, page);
                LOG.infof("Fetching Redfin keyword %s, page %d (request %d)",
                    searchKeyword, page, requestCount + 1);
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("x-api-key", apiKey)
                    .header("Content-Type", "application/json")
                    .GET()
                    .build();
                HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() != 200) {
                    String detail = summarizeError(response.body());
                    String message = "Redfin API returned HTTP " + response.statusCode()
                        + (detail.isBlank() ? "" : ": " + detail);
                    LOG.warnf("Redfin keyword %s page %d failed: %s",
                        searchKeyword, page, message);
                    failure = new FetchFailure(
                        searchKeyword, page, response.statusCode(),
                        message
                    );
                    stop = true;
                    break;
                }

                Map<String, Object> root = mapper.readValue(response.body(),
                    mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
                // The caller persists this full response before this method can
                // advance to another ZIP or page.
                pageHandler.handle(searchKeyword, page, root);
                responses.add(root);
                requestCount++;
                pagesForKeyword++;
                Object pageProperties = root.get("properties");
                if (pageProperties instanceof List<?> values) {
                    for (Object value : values) {
                        if (value instanceof Map<?, ?> map) {
                            properties.add((Map<String, Object>) map);
                        }
                    }
                }

                if (!allPages) break;
                Integer nextPage = nextPage(root.get("pagination"), page);
                if (nextPage == null || requestedPages.contains(nextPage)) break;
                page = nextPage;
            }
            if (stop) break;
        }

        Map<Object, Map<String, Object>> unique = new LinkedHashMap<>();
        for (Map<String, Object> property : properties) {
            Object id = property.get("id");
            Object key = id != null ? id : property.get("addressRaw");
            if (key != null) unique.putIfAbsent(key, property);
        }
        LOG.infof("Redfin fetch %s: %d successful page(s), %d unique listing(s)",
            failure == null ? "complete" : "stopped after failure",
            responses.size(), unique.size());
        return new FetchResult(
            new ArrayList<>(unique.values()), List.copyOf(responses), searchKeywords, failure
        );
    }

    private List<String> configuredKeywords() {
        List<String> result = List.of(keywords.split(",")).stream()
            .map(String::trim)
            .filter(value -> !value.isBlank())
            .distinct()
            .toList();
        if (result.isEmpty()) throw new IllegalStateException("redfin.keywords is empty");
        return result;
    }

    String buildUrl(String searchKeyword, int page) {
        StringBuilder url = new StringBuilder(apiUrl)
            .append("?keyword=").append(encode(searchKeyword))
            .append("&type=").append(encode(listingType))
            .append("&sort=").append(encode(sort));
        for (String homeType : homeTypes.split(",")) {
            if (!homeType.isBlank()) {
                url.append("&homeTypes%5B%5D=").append(encode(homeType.trim()));
            }
        }
        String configuredTimeOnRedfin = timeOnRedfin == null ? "" : timeOnRedfin.orElse("");
        if (!configuredTimeOnRedfin.isBlank()) {
            url.append("&timeOnRedfin=").append(encode(configuredTimeOnRedfin));
        }
        if (page > 1) url.append("&page=").append(page);
        return url.toString();
    }

    private String summarizeError(String body) {
        if (body == null) return "";
        String normalized = body.replaceAll("\\s+", " ").trim();
        return normalized.length() <= 500 ? normalized : normalized.substring(0, 500) + "...";
    }

    private String encode(String value) {
        return URLEncoder.encode(value == null ? "" : value, StandardCharsets.UTF_8);
    }

    private Integer nextPage(Object value, int fallbackCurrentPage) {
        if (!(value instanceof Map<?, ?> pagination)) return null;
        int currentPage = pagination.get("currentPage") instanceof Number number
            ? number.intValue() : fallbackCurrentPage;
        Object next = pagination.get("nextPage");
        if (next != null && !String.valueOf(next).isBlank()) {
            Matcher matcher = PAGE_IN_URL.matcher(String.valueOf(next));
            if (matcher.find()) return Integer.parseInt(matcher.group(1));
            return currentPage + 1;
        }
        Object otherPages = pagination.get("otherPages");
        if (otherPages instanceof Map<?, ?> pages) {
            return pages.keySet().stream()
                .map(String::valueOf)
                .map(key -> {
                    try { return Integer.parseInt(key); }
                    catch (NumberFormatException ignored) { return null; }
                })
                .filter(number -> number != null && number > currentPage)
                .min(Integer::compareTo)
                .orElse(null);
        }
        return null;
    }

    public record FetchResult(
        List<Map<String, Object>> properties,
        List<Map<String, Object>> pageResponses,
        List<String> keywords,
        FetchFailure failure
    ) {
        /** Canonical saved artifact: aggregate properties plus every unmodified API response. */
        public Map<String, Object> savedResponse() {
            Map<String, Object> result = new LinkedHashMap<>();
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("status", failure == null ? "ok" : "partial");
            metadata.put("source", "java-api-fetch");
            metadata.put("provider", "redfin");
            metadata.put("fetchedAt", Instant.now().toString());
            metadata.put("pageCount", pageResponses.size());
            metadata.put("keywords", keywords);
            if (failure != null) metadata.put("failure", failure);
            result.put("requestMetadata", metadata);
            if (!pageResponses.isEmpty()) {
                Object searchInformation = pageResponses.getFirst().get("searchInformation");
                if (searchInformation != null) result.put("searchInformation", searchInformation);
                Object pagination = pageResponses.getLast().get("pagination");
                if (pagination != null) result.put("pagination", pagination);
            }
            result.put("properties", properties);
            result.put("pages", pageResponses);
            return result;
        }
    }

    public record FetchFailure(String keyword, int page, int statusCode, String message) {}

    @FunctionalInterface
    public interface PageHandler {
        void handle(String keyword, int page, Map<String, Object> response) throws Exception;
    }
}
