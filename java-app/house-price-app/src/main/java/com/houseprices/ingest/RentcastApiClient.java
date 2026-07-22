package com.houseprices.ingest;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Calls the RentCast API to fetch recent property sales.
 * Mirrors fetch_rentcast_data.py
 *
 * Config keys (application.properties):
 *   rentcast.api.key
 *   rentcast.lat        (default: 47.60 - Seattle)
 *   rentcast.lng        (default: -122.33)
 *   rentcast.radius     (default: 10 miles)
 *   rentcast.months     (default: 6)
 *   rentcast.limit      (default: 500)
 *   rentcast.timeout.seconds (default: 180)
 */
@ApplicationScoped
public class RentcastApiClient {

    private static final Logger LOG = Logger.getLogger(RentcastApiClient.class);
    private static final String BASE_URL = "https://api.rentcast.io/v1/properties";

    @Inject ObjectMapper mapper;

    @ConfigProperty(name = "rentcast.api.key", defaultValue = "")
    String apiKey;

    @ConfigProperty(name = "rentcast.lat", defaultValue = "47.60")
    double lat;

    @ConfigProperty(name = "rentcast.lng", defaultValue = "-122.33")
    double lng;

    @ConfigProperty(name = "rentcast.radius", defaultValue = "10")
    int radius;

    @ConfigProperty(name = "rentcast.months", defaultValue = "4")
    int months;

    @ConfigProperty(name = "rentcast.limit", defaultValue = "500")
    int limit;

    @ConfigProperty(name = "rentcast.offset", defaultValue = "0")
    int offset;

    @ConfigProperty(name = "rentcast.timeout.seconds", defaultValue = "180")
    int requestTimeoutSeconds;

    private final HttpClient http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .build();

    /**
     * Fetch recent sales from RentCast API.
     * Returns raw property maps ready for DataIngestionService.transformRentcastProperties().
     */
    public List<Map<String, Object>> fetchRecentSales() throws Exception {
        return fetchRecentSales(0);
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> fetchRecentSales(int limitOverride) throws Exception {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("rentcast.api.key is not configured");
        }

        int effectiveLimit = limitOverride > 0 ? limitOverride : limit;
        if (effectiveLimit < 1 || effectiveLimit > 500) {
            throw new IllegalArgumentException("RentCast limit must be between 1 and 500");
        }
        int days = months * 30;
        String url = BASE_URL + "?latitude=" + lat
            + "&longitude=" + lng
            + "&radius=" + radius
            + "&limit=" + effectiveLimit
            + "&offset=" + offset
            + "&includeTotalCount=true"
            + "&saleDateRange=0:" + days
            +"&propertyType=Townhouse%7CSingle%20Family";


        LOG.infof("Fetching RentCast sales: lat=%.2f lng=%.2f"
        +"radius=%d miles days=%d limit=%d",
            lat, lng, radius, days, effectiveLimit);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofSeconds(requestTimeoutSeconds))
            .header("X-Api-Key", apiKey)
            .header("accept", "application/json")
            .GET()
            .build();

        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("RentCast API error " + response.statusCode()
                + ": " + response.body().substring(0, Math.min(200, response.body().length())));
        }

        Object parsed = new com.fasterxml.jackson.databind.ObjectMapper()
            .readValue(response.body(), Object.class);

        List<Map<String, Object>> properties;
        if (parsed instanceof List) {
            properties = (List<Map<String, Object>>) parsed;
        } else if (parsed instanceof Map) {
            Map<String, Object> root = (Map<String, Object>) parsed;
            Object tc = root.get("totalCount");
            if (tc instanceof Number) {
                LOG.infof("RentCast totalCount: %d", ((Number) tc).intValue());
            }
            Object props = root.getOrDefault("properties", root.get("results"));
            properties = props instanceof List ? (List<Map<String, Object>>) props : List.of();
        } else {
            properties = List.of();
        }

        LOG.infof("RentCast returned %d properties", properties.size());
        return properties;
    }
}
