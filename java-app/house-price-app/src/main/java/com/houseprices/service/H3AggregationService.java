package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import com.uber.h3core.H3Core;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Aggregates sales records by H3 L9 hexagon for the map layer.
 * Returns GeoJSON-ready feature data.
 */
@ApplicationScoped
public class H3AggregationService {

    private static final Logger LOG = Logger.getLogger(H3AggregationService.class);

    @Inject PropertyStore store;

    private final H3Core h3;

    public H3AggregationService() {
        try { h3 = H3Core.newInstance(); }
        catch (IOException e) { throw new RuntimeException("H3Core init failed", e); }
    }

    public record HexStats(
        String h3Index,
        double avgSalePrice,
        double avgPredictedPrice,
        double avgPctError,
        double avgSqft,
        int    numSales,
        // Polygon boundary as [[lng, lat], ...]
        List<double[]> boundary
    ) {}

    /**
     * Aggregate all sales records by H3 index with optional filters.
     */
    public List<HexStats> aggregateSales(String variable, String homeType,
                                          String dateFrom, String dateTo) {
        List<PropertyRecord> sales = store.getSalesRecords().stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> homeType == null || homeType.equals("all")
                      || homeType.equalsIgnoreCase(r.homeType()))
            .filter(r -> filterByDate(r, dateFrom, dateTo))
            .collect(Collectors.toList());

        if (sales.isEmpty()) return List.of();

        // Group by H3 index
        Map<String, List<PropertyRecord>> byHex = sales.stream()
            .filter(r -> r.h3Index() != null && !r.h3Index().isBlank())
            .collect(Collectors.groupingBy(PropertyRecord::h3Index));

        List<HexStats> result = new ArrayList<>();
        for (Map.Entry<String, List<PropertyRecord>> entry : byHex.entrySet()) {
            String hexId = entry.getKey();
            List<PropertyRecord> group = entry.getValue();

            double avgSalePrice = group.stream().mapToDouble(PropertyRecord::salePrice).average().orElse(0);
            double avgPredicted = group.stream().mapToDouble(PropertyRecord::predictedPrice).average().orElse(0);
            double avgPctError  = group.stream().mapToDouble(PropertyRecord::pctError).average().orElse(0);
            double avgSqft      = group.stream().mapToDouble(PropertyRecord::sqft).average().orElse(0);

            // Get H3 polygon boundary: list of (lat, lng) -> convert to (lng, lat) for GeoJSON
            List<double[]> boundary;
            try {
                List<com.uber.h3core.util.LatLng> latLngs = h3.cellToBoundary(h3.stringToH3(hexId));
                boundary = latLngs.stream()
                    .map(ll -> new double[]{ll.lng, ll.lat})
                    .collect(Collectors.toList());
                // Close the polygon
                if (!boundary.isEmpty()) boundary.add(boundary.get(0));
            } catch (Exception e) {
                LOG.warnf("Could not get boundary for hex %s: %s", hexId, e.getMessage());
                continue;
            }

            result.add(new HexStats(hexId, avgSalePrice, avgPredicted,
                avgPctError, avgSqft, group.size(), boundary));
        }

        LOG.debugf("Aggregated %d hexes from %d sales records", result.size(), sales.size());
        return result;
    }

    private boolean filterByDate(PropertyRecord r, String from, String to) {
        if ((from == null || from.isBlank()) && (to == null || to.isBlank())) return true;
        if (r.saleDate() == null) return false;
        try {
            if (from != null && !from.isBlank())
                if (r.saleDate().isBefore(java.time.LocalDate.parse(from))) return false;
            if (to != null && !to.isBlank())
                if (r.saleDate().isAfter(java.time.LocalDate.parse(to))) return false;
        } catch (Exception ignored) {}
        return true;
    }

    /** Build a GeoJSON FeatureCollection from aggregated hex stats */    public Map<String, Object> toGeoJson(List<HexStats> hexStats, String variable) {
        List<Map<String, Object>> features = new ArrayList<>();

        for (HexStats hex : hexStats) {
            double displayValue = switch (variable) {
                case "sale_price" -> hex.avgSalePrice();
                case "sqft"       -> hex.avgSqft();
                case "num_sales"  -> hex.numSales();
                default           -> hex.avgPctError();  // pct_error
            };

            Map<String, Object> geometry = Map.of(
                "type", "Polygon",
                "coordinates", List.of(hex.boundary())
            );

            Map<String, Object> props = new LinkedHashMap<>();
            props.put("h3Index",          hex.h3Index());
            props.put("displayValue",     displayValue);
            props.put("avgSalePrice",     hex.avgSalePrice());
            props.put("avgPredictedPrice",hex.avgPredictedPrice());
            props.put("avgPctError",      hex.avgPctError());
            props.put("avgSqft",          hex.avgSqft());
            props.put("numSales",         hex.numSales());

            features.add(Map.of("type", "Feature", "geometry", geometry, "properties", props));
        }

        return Map.of("type", "FeatureCollection", "features", features);
    }
}
