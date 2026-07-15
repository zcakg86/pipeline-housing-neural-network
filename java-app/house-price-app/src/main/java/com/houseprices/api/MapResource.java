package com.houseprices.api;

import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.H3AggregationService;
import com.houseprices.service.PropertyStore;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.io.IOException;
/**
 * REST API for the Leaflet.js map frontend.
 */
@Path("/api")
@Produces(MediaType.APPLICATION_JSON)
public class MapResource {

    @Inject PropertyStore        store;
    @Inject H3AggregationService aggregation;

    /**
     * Sales point layer — serves RentCast, historical sales, or both as GeoJSON points.
     * source: "rentcast" | "sales" | "all"  (default "rentcast")
     * Includes prediction uncertainty and CLS attention weights for variable-driven colouring.
     */
    @GET
    @Path("/rentcast")
    public Map<String, Object> salesPoints(
            @QueryParam("source")    @DefaultValue("rentcast") String source,
            @QueryParam("minError")  @DefaultValue("-500")     double minError,
            @QueryParam("maxError")  @DefaultValue("500")      double maxError,
            @QueryParam("minSqft")   @DefaultValue("0")        double minSqft,
            @QueryParam("maxSqft")   @DefaultValue("10000")    double maxSqft,
            @QueryParam("homeType")  @DefaultValue("all")      String homeType,
            @QueryParam("dateFrom")  @DefaultValue("")         String dateFrom,
            @QueryParam("dateTo")    @DefaultValue("")         String dateTo) {

        // Choose records based on source param
        Stream<PropertyRecord> base = switch (source) {
            case "sales"    -> store.getBySource("sales").stream();
            case "all"      -> Stream.concat(
                                   store.getBySource("rentcast").stream(),
                                   store.getBySource("sales").stream());
            default         -> store.getBySource("rentcast").stream(); // "rentcast"
        };

        List<Map<String, Object>> features = base
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> r.pctError() >= minError && r.pctError() <= maxError)
            .filter(r -> r.sqft() >= minSqft && r.sqft() <= maxSqft)
            .filter(r -> homeType.equals("all") || homeType.equalsIgnoreCase(r.homeType()))
            .filter(r -> filterByDate(r, dateFrom, dateTo))
            .map(r -> {
                Map<String, Object> geom = Map.of(
                    "type", "Point",
                    "coordinates", new double[]{r.lng(), r.lat()}
                );
                Map<String, Object> props = new LinkedHashMap<>();
                props.put("id",                  r.id());
                props.put("address",             r.address());
                props.put("source",              r.source());
                props.put("salePrice",           r.salePrice());
                props.put("predictedPrice",      r.predictedPrice());
                props.put("pctError",            r.pctError());
                props.put("predictionStdPrice",  r.predictionStdPrice());
                props.put("predictionCvPct",     r.predictionCvPct());
                props.put("sqft",                r.sqft());
                props.put("beds",                r.beds());
                props.put("baths",               r.baths());
                props.put("homeType",            r.homeType());
                props.put("saleDate",            r.saleDate() != null ? r.saleDate().toString() : "");
                // CLS attention weights (null-safe)
                float[] a = r.clsAttention();
                if (a != null && a.length == 6) {
                    props.put("attnCommunity", a[0]);
                    props.put("attnYear",      a[1]);
                    props.put("attnWeek",      a[2]);
                    props.put("attnProperty",  a[3]);
                    props.put("attnTime",      a[4]);
                    props.put("attnMarket",    a[5]);
                }
                return Map.of("type", "Feature", "geometry", geom, "properties", props);
            })
            .collect(Collectors.toList());

        return Map.of("type", "FeatureCollection", "features", features);
    }

    /** Zillow listings as GeoJSON points — includes uncertainty and attention for variable colouring */
    @GET
    @Path("/zillow")
    public Map<String, Object> zillowListings(
            @QueryParam("minError")   @DefaultValue("-500")  double minError,
            @QueryParam("maxError")   @DefaultValue("500")   double maxError,
            @QueryParam("minSqft")    @DefaultValue("0")     double minSqft,
            @QueryParam("maxSqft")    @DefaultValue("10000") double maxSqft,
            @QueryParam("homeType")   @DefaultValue("all")   String homeType) {

        List<Map<String, Object>> features = store.getZillowListings().stream()
            .filter(r -> !r.hasSalePrice() || (r.pctError() >= minError && r.pctError() <= maxError))
            .filter(r -> r.sqft() >= minSqft && r.sqft() <= maxSqft)
            .filter(r -> homeType.equals("all") || homeType.equalsIgnoreCase(r.homeType()))
            .map(r -> {
                Map<String, Object> geom = Map.of(
                    "type", "Point",
                    "coordinates", new double[]{r.lng(), r.lat()}
                );
                Map<String, Object> props = new LinkedHashMap<>();
                props.put("id",                  r.id());
                props.put("address",             r.address());
                props.put("listPrice",           r.salePrice());
                props.put("predictedPrice",      r.predictedPrice());
                props.put("pctError",            r.pctError());
                props.put("predictionStdPrice",  r.predictionStdPrice());
                props.put("predictionCvPct",     r.predictionCvPct());
                props.put("sqft",                r.sqft());
                props.put("beds",                r.beds());
                props.put("baths",               r.baths());
                props.put("homeType",            r.homeType());
                props.put("url",                 r.listingUrl());
                float[] a = r.clsAttention();
                if (a != null && a.length == 6) {
                    props.put("attnCommunity", a[0]);
                    props.put("attnYear",      a[1]);
                    props.put("attnWeek",      a[2]);
                    props.put("attnProperty",  a[3]);
                    props.put("attnTime",      a[4]);
                    props.put("attnMarket",    a[5]);
                }
                return Map.of("type", "Feature", "geometry", geom, "properties", props);
            })
            .collect(Collectors.toList());

        return Map.of("type", "FeatureCollection", "features", features);
    }

    /** H3-aggregated sales as GeoJSON polygons */
    @GET
    @Path("/sales/h3")
    public Map<String, Object> salesH3(
            @QueryParam("variable")  @DefaultValue("pct_error") String variable,
            @QueryParam("homeType")  @DefaultValue("all")       String homeType,
            @QueryParam("dateFrom")  @DefaultValue("")          String dateFrom,
            @QueryParam("dateTo")    @DefaultValue("")          String dateTo,
            @QueryParam("minError")  @DefaultValue("-500")      double minError,
            @QueryParam("maxError")  @DefaultValue("500")       double maxError) {

        var hexStats = aggregation.aggregateSales(variable, homeType, dateFrom, dateTo, minError, maxError);
        return aggregation.toGeoJson(hexStats, variable);
    }

    /** Raw sales points for heatmap (lat, lng, pct_error) */
    @GET
    @Path("/sales/points")
    public Map<String, Object> salesPoints(
            @QueryParam("homeType") @DefaultValue("all")  String homeType,
            @QueryParam("dateFrom") @DefaultValue("")     String dateFrom,
            @QueryParam("dateTo")   @DefaultValue("")     String dateTo,
            @QueryParam("minError") @DefaultValue("-500") double minError,
            @QueryParam("maxError") @DefaultValue("500")  double maxError) {

        List<Map<String, Object>> features = store.getSalesRecords().stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> homeType.equals("all") || homeType.equalsIgnoreCase(r.homeType()))
            .filter(r -> filterByDate(r, dateFrom, dateTo))
            .filter(r -> r.pctError() >= minError && r.pctError() <= maxError)
            .map(r -> {
                Map<String, Object> geom = Map.of(
                    "type", "Point",
                    "coordinates", new double[]{r.lng(), r.lat()}
                );
                Map<String, Object> props = new LinkedHashMap<>();
                props.put("pctError",   r.pctError());
                props.put("salePrice",  r.salePrice());
                props.put("sqft",       r.sqft());
                return Map.of("type", "Feature", "geometry", geom, "properties", props);
            })
            .collect(Collectors.toList());

        return Map.of("type", "FeatureCollection", "features", features);
    }

    /** Distinct home types across all sources */
    @GET
    @Path("/home-types")
    public Map<String, Object> homeTypes() {
        Set<String> zillowTypes = store.getZillowListings().stream()
            .map(PropertyRecord::homeType).filter(t -> t != null && !t.isBlank())
            .collect(Collectors.toCollection(TreeSet::new));
        Set<String> salesTypes = store.getSalesRecords().stream()
            .map(PropertyRecord::homeType).filter(t -> t != null && !t.isBlank())
            .collect(Collectors.toCollection(TreeSet::new));
        return Map.of("zillow", zillowTypes, "sales", salesTypes);
    }

    private boolean filterByDate(PropertyRecord r, String from, String to) {
        if ((from == null || from.isBlank()) && (to == null || to.isBlank())) return true;
        if (r.saleDate() == null) return false;
        try {
            if (from != null && !from.isBlank()) {
                if (r.saleDate().isBefore(java.time.LocalDate.parse(from))) return false;
            }
            if (to != null && !to.isBlank()) {
                if (r.saleDate().isAfter(java.time.LocalDate.parse(to))) return false;
            }
        } catch (Exception ignored) {}
        return true;
    }

    /**
     * Quarterly prediction performance by community.
     * Returns top N communities by sale count with avg pct_error and std dev per quarter.
     */
    @GET
    @Path("/performance")
    public Map<String, Object> performance(
            @QueryParam("topN") @DefaultValue("10") int topN) {

        List<PropertyRecord> sales = store.getSalesRecords().stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> r.community() != null && !r.community().isBlank())
            .collect(Collectors.toList());

        if (sales.isEmpty()) return Map.of("communities", List.of(), "series", List.of());

        // Find top N communities by count
        Map<String, Long> countByCommunity = sales.stream()
            .collect(Collectors.groupingBy(PropertyRecord::community, Collectors.counting()));
        List<String> topCommunities = countByCommunity.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(topN)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());

        // Group by community + quarter
        // quarter key: "YYYY-QN"
        Map<String, Map<String, List<Double>>> data = new LinkedHashMap<>();
        for (PropertyRecord r : sales) {
            if (!topCommunities.contains(r.community())) continue;
            if (r.saleDate() == null) continue;
            int year = r.saleDate().getYear();
            int q    = (r.saleDate().getMonthValue() - 1) / 3 + 1;
            String quarter = year + "-Q" + q;
            data.computeIfAbsent(r.community(), k -> new LinkedHashMap<>())
                .computeIfAbsent(quarter, k -> new ArrayList<>())
                .add(r.pctError());
        }

        // Build series per community
        List<Map<String, Object>> series = new ArrayList<>();
        for (String community : topCommunities) {
            Map<String, List<Double>> byQuarter = data.getOrDefault(community, Map.of());
            List<Map<String, Object>> points = new ArrayList<>();
            for (Map.Entry<String, List<Double>> e : new TreeMap<>(byQuarter).entrySet()) {
                List<Double> errors = e.getValue();
                double mean = errors.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double variance = errors.stream()
                    .mapToDouble(v -> (v - mean) * (v - mean)).average().orElse(0);
                double std = Math.sqrt(variance);
                points.add(Map.of(
                    "quarter", e.getKey(),
                    "mean",    mean,
                    "std",     std,
                    "count",   errors.size()
                ));
            }
            series.add(Map.of(
                "community", community,
                "count",     countByCommunity.getOrDefault(community, 0L),
                "points",    points
            ));
        }

        return Map.of("communities", topCommunities, "series", series);
    }

    /**
     * Community layer: H3 L9 hexagons coloured by community ID for the map.
     */
    @GET
    @Path("/sales/community")
    public Map<String, Object> communityLayer() {
        // Collect unique H3 L9 → community from sales records
        Map<String, String> h3ToCommunity = store.getSalesRecords().stream()
            .filter(r -> r.h3Index() != null && r.community() != null && !r.community().isBlank())
            .collect(Collectors.toMap(
                PropertyRecord::h3Index,
                PropertyRecord::community,
                (a, b) -> a
            ));

        Map<String, double[]> communityStats = new HashMap<>();
        for (PropertyRecord r : store.getSalesRecords()) {
            if (r.h3Index() == null || r.community() == null || r.community().isBlank()) continue;
            double[] stats = communityStats.computeIfAbsent(r.community(), k -> new double[6]);
            stats[0] += 1;                       // count
            stats[1] += r.salePrice();
            stats[2] += r.predictedPrice();
            stats[3] += r.pctError();
            stats[4] += r.sqft();
            stats[5] += r.predictionStdPrice();
        }

        List<Map<String, Object>> features = new ArrayList<>();
        try {
            com.uber.h3core.H3Core h3 = com.uber.h3core.H3Core.newInstance();
            for (Map.Entry<String, String> entry : h3ToCommunity.entrySet()) {
                String hexId    = entry.getKey();
                String community = entry.getValue();
                List<com.uber.h3core.util.LatLng> boundary =
                    h3.cellToBoundary(h3.stringToH3(hexId));
                List<double[]> coords = boundary.stream()
                    .map(ll -> new double[]{ll.lng, ll.lat})
                    .collect(Collectors.toList());
                if (!coords.isEmpty()) coords.add(coords.get(0));

                double[] stats = communityStats.getOrDefault(community, new double[6]);
                long salesCount = (long) stats[0];
                double meanSalePrice = salesCount > 0 ? stats[1] / salesCount : 0;
                double meanPredictedPrice = salesCount > 0 ? stats[2] / salesCount : 0;
                double avgPctError = salesCount > 0 ? stats[3] / salesCount : 0;
                double meanSqft = salesCount > 0 ? stats[4] / salesCount : 0;
                double meanPredStd = salesCount > 0 ? stats[5] / salesCount : 0;

                features.add(Map.of(
                    "type", "Feature",
                    "geometry", Map.of("type", "Polygon", "coordinates", List.of(coords)),
                    "properties", Map.of(
                        "community", community,
                        "h3L9", hexId,
                        "salesCount", salesCount,
                        "meanSalePrice", meanSalePrice,
                        "meanPredictedPrice", meanPredictedPrice,
                        "avgPctError", avgPctError,
                        "meanSqft", meanSqft,
                        "meanPredStd", meanPredStd
                    )
                ));
            }
        } catch (Exception e) {
            // return empty on error
        }
        return Map.of("type", "FeatureCollection", "features", features);
    }

    /** Summary statistics */
    @GET
    @Path("/stats")    public Map<String, Object> stats() {
        List<PropertyRecord> zillow = store.getZillowListings();
        List<PropertyRecord> sales  = store.getSalesRecords();
        List<PropertyRecord> rentcast  = store.getRentcastRecords();

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("counts", store.countsBySource());

        if (!zillow.isEmpty()) {
            OptionalDouble avgList = zillow.stream()
                .filter(PropertyRecord::hasSalePrice)
                .mapToDouble(PropertyRecord::salePrice).average();
            OptionalDouble avgPred = zillow.stream()
                .mapToDouble(PropertyRecord::predictedPrice).average();
            result.put("zillow", Map.of(
                "total",        zillow.size(),
                "avgListPrice", avgList.orElse(0),
                "avgPredicted", avgPred.orElse(0)
            ));
        }

        if (!sales.isEmpty()) {
            OptionalDouble avgError = sales.stream()
                .filter(PropertyRecord::hasSalePrice)
                .mapToDouble(r -> Math.abs(r.pctError())).average();
            long uniqueHexes = sales.stream()
                .map(PropertyRecord::h3Index).filter(Objects::nonNull).distinct().count();
            result.put("sales", Map.of(
                "total",       sales.size(),
                "uniqueHexes", uniqueHexes,
                "avgAbsError", avgError.orElse(0)
            ));
        }
            
        if (!rentcast.isEmpty()) {
            OptionalDouble avgError = rentcast.stream()
                .filter(PropertyRecord::hasSalePrice)
                .mapToDouble(r -> Math.abs(r.pctError())).average();
            long uniqueHexes = rentcast.stream()
                .map(PropertyRecord::h3Index).filter(Objects::nonNull).distinct().count();
            long uniqueid = rentcast.stream()
                .map(PropertyRecord::id).filter(Objects::nonNull).distinct().count();
            result.put("rentcast", Map.of(
                "total",       rentcast.size(),
                "uniqueSales", uniqueid,
                "uniqueHexes", uniqueHexes,
                "avgAbsError", avgError.orElse(0)
            ));
        }

        return result;
    }
}
