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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Calls the HasData Zillow scraper API to fetch Seattle listings.
 * Mirrors fetch_zillow_listings.py - supports multi-page fetching.
 *
 * Config keys (application.properties):
 *   zillow.api.key
 *   zillow.keyword      (default: "Seattle, WA")
 *   zillow.type         (default: forSale)
 *   zillow.max.pages    (default: 10)
 *   zillow.delay.ms     (default: 2000)
 */
@ApplicationScoped
public class ZillowApiClient {

    private static final Logger LOG = Logger.getLogger(ZillowApiClient.class);
    private static final String BASE_URL = "https://api.hasdata.com/scrape/zillow/listing";

    @Inject ObjectMapper mapper;

    @ConfigProperty(name = "zillow.api.key", defaultValue = "")
    String apiKey;

    @ConfigProperty(name = "zillow.keyword", defaultValue = "Seattle, WA")
    String keyword;

    @ConfigProperty(name = "zillow.type", defaultValue = "forSale")
    String listingType;

    @ConfigProperty(name = "zillow.max.pages", defaultValue = "10")
    int maxPages;

    @ConfigProperty(name = "zillow.delay.ms", defaultValue = "2000")
    long delayMs;

    private final HttpClient http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .build();

    /**
     * Fetch all pages of Zillow listings, deduplicated by property id.
     * Uses the configured zillow.max.pages by default.
     */
    public List<Map<String, Object>> fetchListings() throws Exception {
        return fetchListings(maxPages);
    }

    /**
     * Fetch with an explicit page limit, overriding the configured default.
     */
    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> fetchListings(int pageLimit) throws Exception {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("zillow.api.key is not configured");
        }

        List<Map<String, Object>> allProperties = new ArrayList<>();
        Integer totalAvailable = null;

        for (int page = 1; page <= pageLimit; page++) {
            String url = BASE_URL
                + "?keyword=" + URLEncoder.encode(keyword, StandardCharsets.UTF_8)
                + "&type=" + listingType
                + "&sort=newest"
                + "&page=" + page
                + "&homeTypes%5B%5D=house"
                + "&homeTypes%5B%5D=townhome";

            LOG.infof("Fetching Zillow page %d/%d...", page, maxPages);

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("x-api-key", apiKey)
                .header("Content-Type", "application/json")
                .GET()
                .build();

            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                LOG.warnf("Zillow API page %d returned %d, stopping", page, response.statusCode());
                break;
            }

            Map<String, Object> root = mapper.readValue(response.body(),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));

            List<Map<String, Object>> pageProps = (List<Map<String, Object>>) root.get("properties");
            if (pageProps == null || pageProps.isEmpty()) {
                LOG.infof("No more properties at page %d, stopping", page);
                break;
            }

            // Capture total on first page
            if (totalAvailable == null) {
                Map<String, Object> searchInfo = (Map<String, Object>) root.get("searchInformation");
                if (searchInfo != null && searchInfo.get("totalResults") instanceof Number n) {
                    totalAvailable = n.intValue();
                    LOG.infof("Total available: %d", totalAvailable);
                }
            }

            allProperties.addAll(pageProps);
            LOG.infof("Page %d: +%d properties (total so far: %d)", page, pageProps.size(), allProperties.size());

            // Stop if we've fetched everything
            if (totalAvailable != null && allProperties.size() >= totalAvailable) {
                LOG.info("Fetched all available properties");
                break;
            }

            // Rate-limit delay (skip after last page)
            if (page < maxPages) {
                Thread.sleep(delayMs);
            }
        }

        // Deduplicate by id
        Map<Object, Map<String, Object>> unique = new java.util.LinkedHashMap<>();
        for (Map<String, Object> prop : allProperties) {
            Object id = prop.get("id");
            if (id != null) unique.putIfAbsent(id, prop);
            else unique.putIfAbsent(prop.get("address"), prop);
        }

        int dupes = allProperties.size() - unique.size();
        if (dupes > 0) LOG.infof("Removed %d duplicates", dupes);

        LOG.infof("Zillow fetch complete: %d unique listings", unique.size());
        return new ArrayList<>(unique.values());
    }
}
