package com.houseprices.api;

import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.H3AggregationService;
import com.houseprices.service.PropertyStore;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;

import java.util.*;
import java.util.stream.Collectors;
/**
 * REST API for the Leaflet.js map frontend.
 */
@Path("/api")
@Produces(MediaType.APPLICATION_JSON)
public class MapResource {

    @Inject PropertyStore        store;
    @Inject H3AggregationService aggregation;

    /** Rentcast recent sales as GeoJSON points */
    @GET
    @Path("/rentcast")
    public Map<String, Object> rentcastPoints(
            @QueryParam("maxError")  @DefaultValue("50")    double maxError,
            @QueryParam("minSqft")   @DefaultValue("0")     double minSqft,
            @QueryParam("maxSqft")   @DefaultValue("10000") double maxSqft,
            @QueryParam("homeType")  @DefaultValue("all")   String homeType) {

        List<Map<String, Object>> features = store.getBySource("rentcast").stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> Math.abs(r.pctError()) <= maxError)
            .filter(r -> r.sqft() >= minSqft && r.sqft() <= maxSqft)
            .filter(r -> homeType.equals("all") || homeType.equalsIgnoreCase(r.homeType()))
            .map(r -> {
                Map<String, Object> geom = Map.of(
                    "type", "Point",
                    "coordinates", new double[]{r.lng(), r.lat()}
                );
                Map<String, Object> props = new LinkedHashMap<>();
                props.put("id",             r.id());
                props.put("address",        r.address());
                props.put("salePrice",      r.salePrice());
                props.put("predictedPrice", r.predictedPrice());
                props.put("pctError",       r.pctError());
                props.put("sqft",           r.sqft());
                props.put("beds",           r.beds());
                props.put("baths",          r.baths());
                props.put("homeType",       r.homeType());
                props.put("saleDate",       r.saleDate() != null ? r.saleDate().toString() : "");
                return Map.of("type", "Feature", "geometry", geom, "properties", props);
            })
            .collect(Collectors.toList());

        return Map.of("type", "FeatureCollection", "features", features);
    }

    /** Zillow listings as GeoJSON points */
    @GET
    @Path("/zillow")
    public Map<String, Object> zillowListings(
            @QueryParam("maxError")   @DefaultValue("50")    double maxError,
            @QueryParam("minSqft")    @DefaultValue("0")     double minSqft,
            @QueryParam("maxSqft")    @DefaultValue("10000") double maxSqft,
            @QueryParam("homeType")   @DefaultValue("all")   String homeType) {

        List<Map<String, Object>> features = store.getZillowListings().stream()
            .filter(r -> Math.abs(r.pctError()) <= maxError || !r.hasSalePrice())
            .filter(r -> r.sqft() >= minSqft && r.sqft() <= maxSqft)
            .filter(r -> homeType.equals("all") || homeType.equalsIgnoreCase(r.homeType()))
            .map(r -> {
                Map<String, Object> geom = Map.of(
                    "type", "Point",
                    "coordinates", new double[]{r.lng(), r.lat()}
                );
                Map<String, Object> props = new LinkedHashMap<>();
                props.put("id",             r.id());
                props.put("address",        r.address());
                props.put("listPrice",      r.salePrice());
                props.put("predictedPrice", r.predictedPrice());
                props.put("pctError",       r.pctError());
                props.put("sqft",           r.sqft());
                props.put("beds",           r.beds());
                props.put("baths",          r.baths());
                props.put("homeType",       r.homeType());
                props.put("url",            r.listingUrl());
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
            @QueryParam("dateTo")    @DefaultValue("")          String dateTo) {

        var hexStats = aggregation.aggregateSales(variable, homeType, dateFrom, dateTo);
        return aggregation.toGeoJson(hexStats, variable);
    }

    /** Raw sales points for heatmap (lat, lng, pct_error) */
    @GET
    @Path("/sales/points")
    public Map<String, Object> salesPoints(
            @QueryParam("homeType") @DefaultValue("all") String homeType,
            @QueryParam("dateFrom") @DefaultValue("")    String dateFrom,
            @QueryParam("dateTo")   @DefaultValue("")    String dateTo) {

        List<Map<String, Object>> features = store.getSalesRecords().stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> homeType.equals("all") || homeType.equalsIgnoreCase(r.homeType()))
            .filter(r -> filterByDate(r, dateFrom, dateTo))
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

    /** Summary statistics */
    @GET
    @Path("/stats")
    public Map<String, Object> stats() {
        List<PropertyRecord> zillow = store.getZillowListings();
        List<PropertyRecord> sales  = store.getSalesRecords();

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

        return result;
    }
}
