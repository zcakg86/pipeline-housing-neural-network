package com.houseprices.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.uber.h3core.H3Core;
import com.uber.h3core.util.LatLng;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** Calculates distance-decayed water proximity from the exported geometry. */
@ApplicationScoped
public class WaterProximityService {

    public static final double WATER_PROXIMITY_TAU_METERS = 100.0;
    private static final Logger LOG = Logger.getLogger(WaterProximityService.class);
    private static final double EARTH_RADIUS_METERS = 6_371_008.8;
    private static final double REFERENCE_LATITUDE_RADIANS = Math.toRadians(47.4);
    private static final double LONGITUDE_SCALE =
        EARTH_RADIUS_METERS * Math.cos(REFERENCE_LATITUDE_RADIANS) * Math.PI / 180.0;
    private static final double LATITUDE_SCALE = EARTH_RADIUS_METERS * Math.PI / 180.0;
    private static final double BUCKET_SIZE_METERS = 2_000.0;

    public record WaterFeatures(double distanceToWaterM) {}

    public static double waterProximity(double distanceToWaterM) {
        return Math.exp(-Math.max(0.0, distanceToWaterM) / WATER_PROXIMITY_TAU_METERS);
    }

    private record Segment(double x1, double y1, double x2, double y2) {}
    private record PointKey(long latitudeBits, long longitudeBits) {}

    private final List<Segment> segments = new ArrayList<>();
    private final Map<Long, List<Integer>> buckets = new HashMap<>();
    private final Map<PointKey, WaterFeatures> cache = new ConcurrentHashMap<>();
    private H3Core h3;

    @PostConstruct
    void load() {
        try {
            h3 = H3Core.newInstance();
            ObjectMapper mapper = new ObjectMapper();
            try (InputStream input = getClass().getClassLoader()
                    .getResourceAsStream("model-artifacts/king_county_water.geojson")) {
                if (input == null) {
                    throw new IllegalStateException(
                        "king_county_water.geojson not found in model artifacts"
                    );
                }
                JsonNode root = mapper.readTree(input);
                for (JsonNode feature : root.path("features")) {
                    addGeometry(feature.path("geometry"));
                }
            }
            if (segments.isEmpty()) {
                throw new IllegalStateException("Water artifact contains no boundary segments");
            }
            LOG.infof(
                "Loaded %,d water-boundary segments into %,d spatial buckets",
                segments.size(), buckets.size()
            );
        } catch (Exception e) {
            throw new RuntimeException("Failed to load water proximity artifact", e);
        }
    }

    public WaterFeatures lookup(double latitude, double longitude) {
        if (!Double.isFinite(latitude) || !Double.isFinite(longitude)) {
            throw new IllegalArgumentException("Finite latitude and longitude are required");
        }
        PointKey key = new PointKey(
            Double.doubleToLongBits(latitude), Double.doubleToLongBits(longitude)
        );
        return cache.computeIfAbsent(key, unused -> {
            double x = longitude * LONGITUDE_SCALE;
            double y = latitude * LATITUDE_SCALE;
            double distance = nearestDistance(x, y);
            return new WaterFeatures(distance);
        });
    }

    public WaterFeatures lookupH3(String h3Index) {
        try {
            LatLng center = h3.cellToLatLng(h3.stringToH3(h3Index));
            return lookup(center.lat, center.lng);
        } catch (Exception e) {
            throw new IllegalArgumentException("Cannot resolve H3 center: " + h3Index, e);
        }
    }

    private double nearestDistance(double x, double y) {
        int centerX = bucket(x);
        int centerY = bucket(y);
        double best = Double.POSITIVE_INFINITY;
        Set<Integer> visited = new HashSet<>();

        for (int radius = 0; radius <= 100; radius++) {
            for (int bx = centerX - radius; bx <= centerX + radius; bx++) {
                checkBucket(bx, centerY - radius, x, y, visited);
                if (radius > 0) checkBucket(bx, centerY + radius, x, y, visited);
            }
            for (int by = centerY - radius + 1; by < centerY + radius; by++) {
                checkBucket(centerX - radius, by, x, y, visited);
                if (radius > 0) checkBucket(centerX + radius, by, x, y, visited);
            }
            for (int index : visited) {
                best = Math.min(best, pointToSegmentDistance(x, y, segments.get(index)));
            }

            double left = (centerX - radius) * BUCKET_SIZE_METERS;
            double right = (centerX + radius + 1.0) * BUCKET_SIZE_METERS;
            double bottom = (centerY - radius) * BUCKET_SIZE_METERS;
            double top = (centerY + radius + 1.0) * BUCKET_SIZE_METERS;
            double distanceToUnsearched = Math.min(
                Math.min(x - left, right - x), Math.min(y - bottom, top - y)
            );
            if (best <= distanceToUnsearched) return best;
        }
        if (Double.isFinite(best)) return best;
        throw new IllegalStateException("No water boundary found within 200 km of prediction");
    }

    private void checkBucket(
        int x, int y, double pointX, double pointY, Set<Integer> visited
    ) {
        List<Integer> values = buckets.get(bucketKey(x, y));
        if (values != null) visited.addAll(values);
    }

    private static double pointToSegmentDistance(double x, double y, Segment segment) {
        double dx = segment.x2() - segment.x1();
        double dy = segment.y2() - segment.y1();
        double lengthSquared = dx * dx + dy * dy;
        if (lengthSquared == 0.0) return Math.hypot(x - segment.x1(), y - segment.y1());
        double position = ((x - segment.x1()) * dx + (y - segment.y1()) * dy)
            / lengthSquared;
        position = Math.max(0.0, Math.min(1.0, position));
        return Math.hypot(
            x - (segment.x1() + position * dx),
            y - (segment.y1() + position * dy)
        );
    }

    private void addGeometry(JsonNode geometry) {
        String type = geometry.path("type").asText();
        JsonNode coordinates = geometry.path("coordinates");
        switch (type) {
            case "LineString" -> addLine(coordinates);
            case "MultiLineString", "Polygon" -> coordinates.forEach(this::addLine);
            case "MultiPolygon" -> coordinates.forEach(
                polygon -> polygon.forEach(this::addLine)
            );
            case "GeometryCollection" -> geometry.path("geometries").forEach(this::addGeometry);
            default -> { /* Point features do not define a water boundary. */ }
        }
    }

    private void addLine(JsonNode coordinates) {
        for (int i = 1; i < coordinates.size(); i++) {
            JsonNode first = coordinates.get(i - 1);
            JsonNode second = coordinates.get(i);
            addSegment(
                first.get(0).asDouble() * LONGITUDE_SCALE,
                first.get(1).asDouble() * LATITUDE_SCALE,
                second.get(0).asDouble() * LONGITUDE_SCALE,
                second.get(1).asDouble() * LATITUDE_SCALE
            );
        }
    }

    private void addSegment(double x1, double y1, double x2, double y2) {
        int index = segments.size();
        segments.add(new Segment(x1, y1, x2, y2));
        int minX = bucket(Math.min(x1, x2));
        int maxX = bucket(Math.max(x1, x2));
        int minY = bucket(Math.min(y1, y2));
        int maxY = bucket(Math.max(y1, y2));
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                buckets.computeIfAbsent(bucketKey(x, y), unused -> new ArrayList<>()).add(index);
            }
        }
    }

    private static int bucket(double coordinate) {
        return (int) Math.floor(coordinate / BUCKET_SIZE_METERS);
    }

    private static long bucketKey(int x, int y) {
        return ((long) x << 32) ^ (y & 0xffffffffL);
    }
}
