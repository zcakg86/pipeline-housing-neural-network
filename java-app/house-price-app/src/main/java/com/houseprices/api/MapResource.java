package com.houseprices.api;

import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.H3AggregationService;
import com.houseprices.service.LiveSourceLoader;
import com.houseprices.service.PropertyRequestFilter;
import com.houseprices.service.PropertyStore;
import com.houseprices.service.PropertyTimeSeriesService;
import com.houseprices.service.SyntheticPredictionService;
import com.houseprices.service.TransportFeatureService;
import com.houseprices.model.RentcastLocalMarketService;
import com.houseprices.model.WaterProximityService;
import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.LightGBMModel;
import com.houseprices.model.GnnModel;
import com.houseprices.model.MarketIndicatorService;
import com.houseprices.model.ModelArtifacts;
import com.houseprices.model.PredictionContext;
import com.houseprices.model.PredictionContextFactory;
import com.houseprices.model.FeatureContract;
import com.uber.h3core.H3Core;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.stream.Collectors;
import java.io.IOException;
/**
 * REST API for the Leaflet.js map frontend.
 */
@Path("/api")
@Produces(MediaType.APPLICATION_JSON)
public class MapResource {

    private static final Logger LOG = Logger.getLogger(MapResource.class);
    private static final int DEFAULT_POINT_LIMIT = 5_000;
    private static final int MAX_POINT_LIMIT = 20_000;
    private static final int RAW_POINT_MIN_ZOOM = 13;
    private final H3Core h3;

    public MapResource() {
        try { h3 = H3Core.newInstance(); }
        catch (IOException exception) { throw new RuntimeException("H3Core init failed", exception); }
    }

    @Inject PropertyStore        store;
    @Inject H3AggregationService aggregation;
    @Inject SyntheticPredictionService syntheticPredictions;
    @Inject WaterProximityService waterProximity;
    @Inject EmbeddingModel neuralModel;
    @Inject LightGBMModel lightgbmModel;
    @Inject GnnModel gnnModel;
    @Inject MarketIndicatorService marketIndicators;
    @Inject ModelArtifacts modelArtifacts;
    @Inject PredictionContextFactory contextFactory;
    @Inject PropertyTimeSeriesService propertyTimeSeries;
    @Inject TransportFeatureService transportFeatures;
    @Inject RentcastLocalMarketService rentcastLocalMarket;
    @Inject LiveSourceLoader liveSources;

    /** Receives concise browser interaction events for the operational log. */
    @POST
    @Path("/events")
    @Consumes(MediaType.APPLICATION_JSON)
    public Map<String, String> logUiEvent(Map<String, Object> event) {
        String action = String.valueOf(event.getOrDefault("action", "unknown"));
        String detail = String.valueOf(event.getOrDefault("detail", ""));
        LOG.infof("Map UI event: %s — %s", action.substring(0, Math.min(action.length(), 120)),
            detail.substring(0, Math.min(detail.length(), 240)));
        return Map.of("status", "logged");
    }

    /** Static OSM-derived H3 accessibility fields for geographic inspection. */
    @GET
    @Path("/transport-features")
    public Map<String, Object> transportFeatures() {
        return transportFeatures.featureCollection();
    }

    /** File-backed H3 L9 synthetic properties, recalculated for the requested date. */
    @GET
    @Path("/synthetic")
    public Map<String, Object> syntheticLayer(
            @QueryParam("saleDate") @DefaultValue("") String saleDate) {
        try {
            java.time.LocalDate date = parsePredictionDate(saleDate);
            return syntheticPredictions.predict(date);
        } catch (IllegalArgumentException exception) {
            throw new BadRequestException(exception.getMessage());
        }
    }

    @GET
    @Path("/synthetic/meta")
    public Map<String, Object> syntheticMetadata() {
        return Map.of(
            "count", syntheticPredictions.size(),
            "defaultSaleDate", java.time.LocalDate.now().toString(),
            "minimumSaleDate", syntheticPredictions.minimumPredictionDate().toString(),
            "monthStep", 1
        );
    }

    /** Exact LightGBM plus exact-group and sampled-feature neural Shapley effects. */
    @GET
    @Path("/synthetic/explanation")
    public Map<String, Object> syntheticExplanation(
            @QueryParam("h3Index") @DefaultValue("") String h3Index,
            @QueryParam("saleDate") @DefaultValue("") String saleDate,
            @QueryParam("lat") Double latitude,
            @QueryParam("lng") Double longitude) {
        if (h3Index == null || h3Index.isBlank()) {
            throw new BadRequestException("h3Index is required");
        }
        try {
            java.time.LocalDate date = parsePredictionDate(saleDate);
            return syntheticPredictions.explain(
                h3Index, date, latitude, longitude
            );
        } catch (IllegalArgumentException exception) {
            throw new BadRequestException(exception.getMessage());
        }
    }

    /** Fast prediction for one synthetic cell at user-adjusted coordinates. */
    @GET
    @Path("/synthetic/point")
    public Map<String, Object> syntheticPoint(
            @QueryParam("h3Index") @DefaultValue("") String h3Index,
            @QueryParam("saleDate") @DefaultValue("") String saleDate,
            @QueryParam("lat") Double latitude,
            @QueryParam("lng") Double longitude) {
        if (h3Index == null || h3Index.isBlank()) {
            throw new BadRequestException("h3Index is required");
        }
        try {
            return syntheticPredictions.predictPoint(
                h3Index, parsePredictionDate(saleDate), latitude, longitude
            );
        } catch (IllegalArgumentException exception) {
            throw new BadRequestException(exception.getMessage());
        }
    }

    private static java.time.LocalDate parsePredictionDate(String value) {
        if (value == null || value.isBlank()) return java.time.LocalDate.now();
        try {
            return java.time.LocalDate.parse(value);
        } catch (java.time.format.DateTimeParseException exception) {
            throw new IllegalArgumentException(
                "saleDate must use ISO format YYYY-MM-DD", exception
            );
        }
    }

    /**
     * Feature contributions for one stored Zillow, RentCast, or historical-sale point.
     * LightGBM and neural results are feature-level; GNN results are exact effects
     * for its deployed graph/property/time/economic input groups.
     */
    @GET
    @Path("/points/explanation")
    public Map<String, Object> pointExplanation(
            @QueryParam("source") @DefaultValue("") String source,
            @QueryParam("id") @DefaultValue("") String id) {
        if (!Set.of("zillow", "rentcast", "sales").contains(source)) {
            throw new BadRequestException("source must be zillow, rentcast, or sales");
        }
        if (id == null || id.isBlank()) throw new BadRequestException("id is required");
        PropertyRecord record = store.findBySourceAndId(source, id)
            .orElseThrow(() -> new NotFoundException("Property record not found"));
        java.time.LocalDate date = record.saleDate() == null
            ? java.time.LocalDate.now() : record.saleDate();

        // Historical CSV predictions used their row-specific economic values.
        // Live Zillow/RentCast ingestion currently uses the deployment defaults.
        double mortgageRate = FeatureContract.DEFAULT_MORTGAGE_RATE;
        double unemploymentRate = FeatureContract.DEFAULT_UNEMPLOYMENT_RATE;
        String indicatorDate = "deployment defaults";
        if ("sales".equals(source)) {
            MarketIndicatorService.MarketIndicators market = marketIndicators.lookup(date);
            mortgageRate = market.mortgageRate();
            unemploymentRate = market.unemploymentRate();
            indicatorDate = market.effectiveDate().toString();
        }
        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            record.h3Index(), date, record.sqft(), record.sqftLot(), record.beds(),
            record.lat(), record.lng(), mortgageRate, unemploymentRate
        );
        java.time.LocalDate latestSnapshotSale = modelArtifacts.getLocalMarketLatestSaleDate();
        boolean snapshotLookAhead = latestSnapshotSale != null && !date.isAfter(latestSnapshotSale);
        PredictionContext context = "rentcast".equals(source)
            ? contextFactory.prepareRentcast(input)
            : contextFactory.prepare(input, snapshotLookAhead);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("source", source);
        result.put("id", id);
        result.put("address", record.address());
        result.put("h3Index", record.h3Index());
        result.put("saleDate", date.toString());
        result.put("mortgageRate", mortgageRate);
        result.put("unemploymentRate", unemploymentRate);
        result.put("marketIndicatorDate", indicatorDate);
        result.put("storedNeuralPrediction", record.predictedPrice());
        result.put("storedLightgbmPrediction", record.lightgbmPredictedPrice());
        result.put("storedGnnPrediction", record.gnnPredictedPrice());
        result.put("historicalSnapshotReconstruction", snapshotLookAhead);
        if ("rentcast".equals(source)) {
            result.put("localMarketPredictionIssue",
                rentcastLocalMarket.hasLookAheadIssue(date) ? "rolling_snapshot_look_ahead" : "");
            result.put("rentcastLocalMarketLatestSaleDate",
                rentcastLocalMarket.latestSaleDate().toString());
        }
        result.put("explanation", lightgbmModel.explainPrepared(context));
        result.put("neuralExplanation", neuralModel.explainPrepared(context));
        result.put("gnnExplanation", gnnModel.explain(new GnnModel.Input(
            record.h3Index(), date, record.sqft(), record.sqftLot(), record.beds(),
            record.lat(), record.lng()
        )));
        return result;
    }

    /** Monthly counterfactual value history for one stored map observation. */
    @GET
    @Path("/points/time-series")
    public Map<String, Object> pointTimeSeries(
            @QueryParam("source") @DefaultValue("") String source,
            @QueryParam("id") @DefaultValue("") String id) {
        if (!Set.of("zillow", "rentcast", "sales").contains(source)) {
            throw new BadRequestException("source must be zillow, rentcast, or sales");
        }
        if (id == null || id.isBlank()) throw new BadRequestException("id is required");
        try {
            return propertyTimeSeries.predict(source, id);
        } catch (IllegalArgumentException exception) {
            throw new NotFoundException(exception.getMessage());
        }
    }

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
            @QueryParam("dateTo")    @DefaultValue("")         String dateTo,
            @QueryParam("model")     @DefaultValue("lightgbm") String model,
            @QueryParam("west") Double west,
            @QueryParam("south") Double south,
            @QueryParam("east") Double east,
            @QueryParam("north") Double north,
            @QueryParam("zoom") @DefaultValue("12") int zoom,
            @QueryParam("limit") @DefaultValue("5000") int limit) {

        if (!"sales".equals(source)) liveSources.ensureLoaded("rentcast");

        PropertyRequestFilter filter = requestFilter(
            homeType, dateFrom, dateTo, minError, maxError, west, south, east, north
        );
        PropertyStore.Snapshot snapshot = store.snapshot();
        List<PropertyRecord> base = switch (source) {
            case "sales" -> snapshot.sales();
            case "all" -> snapshot.completedSales();
            default -> snapshot.rentcast();
        };
        List<PropertyRecord> matches = base.stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> r.sqft() >= minSqft && r.sqft() <= maxSqft)
            .filter(r -> filter.includes(r, selectedError(r, model)))
            .toList();
        if (zoom < RAW_POINT_MIN_ZOOM) {
            List<Map<String, Object>> features = detailedPointClusters(matches, zoom);
            return pointCollection(features, matches.size(), false, true, zoom);
        }

        int effectiveLimit = validatedLimit(limit);
        boolean truncated = matches.size() > effectiveLimit;
        List<Map<String, Object>> features = matches.stream().limit(effectiveLimit)
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
                props.put("lightgbmPredictedPrice", r.lightgbmPredictedPrice());
                props.put("lightgbmPctError", r.lightgbmPctError());
                props.put("gnnPredictedPrice", r.gnnPredictedPrice());
                props.put("gnnPctError", r.gnnPctError());
                props.put("predictionStdPrice",  r.predictionStdPrice());
                props.put("predictionCvPct",     r.predictionCvPct());
                props.put("sqft",                r.sqft());
                props.put("sqftLot",             r.sqftLot());
                props.put("beds",                r.beds());
                props.put("baths",               r.baths());
                props.put("homeType",            r.homeType());
                props.put("saleDate",            r.saleDate() != null ? r.saleDate().toString() : "");
                putRentcastPredictionStatus(props, r);
                putWaterFeatures(props, r);
                putPredictionIntervals(props, r);
                // CLS attention weights (null-safe)
                float[] a = r.clsAttention();
                if (a != null && a.length == 4) {
                    props.put("attnCommunity", a[0]);
                    props.put("attnProperty",  a[1]);
                    props.put("attnTime",      a[2]);
                    props.put("attnMarket",    a[3]);
                }
                return Map.of("type", "Feature", "geometry", geom, "properties", props);
            })
            .collect(Collectors.toList());

        return pointCollection(features, matches.size(), truncated, false, zoom);
    }

    /** Zillow listings as GeoJSON points — includes uncertainty and attention for variable colouring */
    @GET
    @Path("/zillow")
    public Map<String, Object> zillowListings(
            @QueryParam("minError")   @DefaultValue("-500")  double minError,
            @QueryParam("maxError")   @DefaultValue("500")   double maxError,
            @QueryParam("minSqft")    @DefaultValue("0")     double minSqft,
            @QueryParam("maxSqft")    @DefaultValue("10000") double maxSqft,
            @QueryParam("homeType")   @DefaultValue("all")   String homeType,
            @QueryParam("model")      @DefaultValue("lightgbm") String model,
            @QueryParam("west") Double west,
            @QueryParam("south") Double south,
            @QueryParam("east") Double east,
            @QueryParam("north") Double north,
            @QueryParam("zoom") @DefaultValue("12") int zoom,
            @QueryParam("limit") @DefaultValue("5000") int limit) {

        liveSources.ensureLoaded("zillow");

        PropertyRequestFilter filter = requestFilter(
            homeType, "", "", minError, maxError, west, south, east, north
        );
        List<PropertyRecord> matches = store.snapshot().zillow().stream()
            .filter(r -> r.sqft() >= minSqft && r.sqft() <= maxSqft)
            .filter(r -> filter.includes(r, selectedError(r, model)))
            .toList();
        if (zoom < RAW_POINT_MIN_ZOOM) {
            List<Map<String, Object>> features = detailedPointClusters(matches, zoom);
            return pointCollection(features, matches.size(), false, true, zoom);
        }
        int effectiveLimit = validatedLimit(limit);
        boolean truncated = matches.size() > effectiveLimit;
        List<Map<String, Object>> features = matches.stream().limit(effectiveLimit)
            .map(r -> {
                Map<String, Object> geom = Map.of(
                    "type", "Point",
                    "coordinates", new double[]{r.lng(), r.lat()}
                );
                Map<String, Object> props = new LinkedHashMap<>();
                props.put("id",                  r.id());
                props.put("address",             r.address());
                props.put("source",              r.source());
                props.put("listPrice",           r.salePrice());
                props.put("zestimate",           r.zestimate());
                props.put("predictedPrice",      r.predictedPrice());
                props.put("pctError",            r.pctError());
                props.put("lightgbmPredictedPrice", r.lightgbmPredictedPrice());
                props.put("lightgbmPctError", r.lightgbmPctError());
                props.put("gnnPredictedPrice", r.gnnPredictedPrice());
                props.put("gnnPctError", r.gnnPctError());
                props.put("predictionStdPrice",  r.predictionStdPrice());
                props.put("predictionCvPct",     r.predictionCvPct());
                props.put("sqft",                r.sqft());
                props.put("sqftLot",             r.sqftLot());
                props.put("beds",                r.beds());
                props.put("baths",               r.baths());
                props.put("homeType",            r.homeType());
                props.put("url",                 r.listingUrl());
                props.put("saleDate",            r.saleDate() != null ? r.saleDate().toString() : "");
                props.put("predictionDate",      r.saleDate() != null ? r.saleDate().toString() : "");
                putWaterFeatures(props, r);
                putPredictionIntervals(props, r);
                float[] a = r.clsAttention();
                if (a != null && a.length == 4) {
                    props.put("attnCommunity", a[0]);
                    props.put("attnProperty",  a[1]);
                    props.put("attnTime",      a[2]);
                    props.put("attnMarket",    a[3]);
                }
                return Map.of("type", "Feature", "geometry", geom, "properties", props);
            })
            .collect(Collectors.toList());

        return pointCollection(features, matches.size(), truncated, false, zoom);
    }

    /** H3-aggregated sales as GeoJSON polygons */
    @GET
    @Path("/sales/h3")
    public Map<String, Object> salesH3(
            @QueryParam("variable")  @DefaultValue("pct_error") String variable,
            @QueryParam("model")     @DefaultValue("neural")    String model,
            @QueryParam("homeType")  @DefaultValue("all")       String homeType,
            @QueryParam("dateFrom")  @DefaultValue("")          String dateFrom,
            @QueryParam("dateTo")    @DefaultValue("")          String dateTo,
            @QueryParam("minError")  @DefaultValue("-500")      double minError,
            @QueryParam("maxError")  @DefaultValue("500")       double maxError) {

        liveSources.ensureLoaded("sales");

        String effectiveModel = "predicted_price_lightgbm".equals(variable)
            ? "lightgbm"
            : "predicted_price_neural".equals(variable) ? "neural" : model;
        try {
            PropertyRequestFilter filter = PropertyRequestFilter.parse(
                homeType, dateFrom, dateTo, minError, maxError,
                null, null, null, null
            );
            var hexStats = aggregation.aggregateSales(
                variable, effectiveModel, filter
            );
            return aggregation.toGeoJson(hexStats, variable, effectiveModel);
        } catch (IllegalArgumentException exception) {
            throw new BadRequestException(exception.getMessage());
        }
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

    private PropertyRequestFilter requestFilter(
            String homeType,
            String dateFrom,
            String dateTo,
            double minError,
            double maxError,
            Double west,
            Double south,
            Double east,
            Double north) {
        try {
            return PropertyRequestFilter.parse(
                homeType, dateFrom, dateTo, minError, maxError,
                west, south, east, north
            );
        } catch (IllegalArgumentException exception) {
            throw new BadRequestException(exception.getMessage());
        }
    }

    private int validatedLimit(int limit) {
        if (limit < 1) throw new BadRequestException("limit must be positive");
        return Math.min(limit, MAX_POINT_LIMIT);
    }

    private Map<String, Object> pointCollection(
            List<Map<String, Object>> features, int matched, boolean truncated,
            boolean clustered, int zoom) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("type", "FeatureCollection");
        result.put("matched", matched);
        result.put("returned", features.size());
        result.put("truncated", truncated);
        result.put("clustered", clustered);
        result.put("zoom", zoom);
        result.put("features", features);
        return result;
    }

    private int clusterResolution(int zoom) {
        if (zoom <= 9) return 6;
        if (zoom <= 11) return 7;
        return 8;
    }

    private Map<String, PointAccumulator> cluster(List<PropertyRecord> records, int zoom) {
        int resolution = clusterResolution(zoom);
        Map<String, PointAccumulator> clusters = new LinkedHashMap<>();
        for (PropertyRecord record : records) {
            String cell;
            try {
                long original = h3.latLngToCell(record.lat(), record.lng(), resolution);
                cell = h3.h3ToString(original);
            } catch (Exception exception) {
                continue;
            }
            clusters.computeIfAbsent(cell, ignored -> new PointAccumulator()).add(record);
        }
        return clusters;
    }

    private List<Map<String, Object>> detailedPointClusters(
            List<PropertyRecord> records, int zoom) {
        List<Map<String, Object>> features = new ArrayList<>();
        for (Map.Entry<String, PointAccumulator> entry : cluster(records, zoom).entrySet()) {
            PointAccumulator values = entry.getValue();
            Map<String, Object> properties = values.properties();
            properties.put("id", entry.getKey());
            properties.put("source", "cluster");
            properties.put("cluster", true);
            properties.put("count", values.count);
            features.add(pointFeature(values.avgLng(), values.avgLat(), properties));
        }
        return features;
    }

    private Map<String, Object> pointFeature(
            double longitude, double latitude, Map<String, Object> properties) {
        return Map.of(
            "type", "Feature",
            "geometry", Map.of("type", "Point", "coordinates", new double[]{longitude, latitude}),
            "properties", properties
        );
    }

    private static final class PointAccumulator {
        int count;
        int attentionCount;
        double lat;
        double lng;
        double salePrice;
        double zestimate;
        int zestimateCount;
        double neuralPrediction;
        double treePrediction;
        double gnnPrediction;
        double neuralError;
        double treeError;
        double gnnError;
        double predictionStd;
        double predictionCv;
        double sqft;
        double sqftLot;
        double beds;
        double baths;
        final double[] attention = new double[4];

        void add(PropertyRecord record) {
            count++;
            lat += record.lat();
            lng += record.lng();
            salePrice += record.salePrice();
            if (record.zestimate() > 0) {
                zestimate += record.zestimate();
                zestimateCount++;
            }
            neuralPrediction += record.predictedPrice();
            treePrediction += record.lightgbmPredictedPrice();
            gnnPrediction += record.gnnPredictedPrice();
            neuralError += record.pctError();
            treeError += record.lightgbmPctError();
            gnnError += record.gnnPctError();
            predictionStd += record.predictionStdPrice();
            predictionCv += record.predictionCvPct();
            sqft += record.sqft();
            sqftLot += record.sqftLot();
            beds += record.beds();
            baths += record.baths();
            float[] weights = record.clsAttention();
            if (weights != null && weights.length == 4) {
                attentionCount++;
                for (int index = 0; index < 4; index++) attention[index] += weights[index];
            }
        }

        double avgLat() { return lat / count; }
        double avgLng() { return lng / count; }

        Map<String, Object> properties() {
            double divisor = count;
            double attentionDivisor = Math.max(1, attentionCount);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("salePrice", salePrice / divisor);
            result.put("listPrice", salePrice / divisor);
            result.put("zestimate", zestimateCount > 0 ? zestimate / zestimateCount : 0.0);
            result.put("predictedPrice", neuralPrediction / divisor);
            result.put("lightgbmPredictedPrice", treePrediction / divisor);
            result.put("gnnPredictedPrice", gnnPrediction / divisor);
            result.put("pctError", neuralError / divisor);
            result.put("lightgbmPctError", treeError / divisor);
            result.put("gnnPctError", gnnError / divisor);
            result.put("predictionStdPrice", predictionStd / divisor);
            result.put("predictionCvPct", predictionCv / divisor);
            result.put("sqft", sqft / divisor);
            result.put("sqftLot", sqftLot / divisor);
            result.put("beds", beds / divisor);
            result.put("baths", baths / divisor);
            result.put("attnCommunity", attention[0] / attentionDivisor);
            result.put("attnProperty", attention[1] / attentionDivisor);
            result.put("attnTime", attention[2] / attentionDivisor);
            result.put("attnMarket", attention[3] / attentionDivisor);
            return result;
        }
    }

    private double selectedError(PropertyRecord record, String model) {
        return switch (model.toLowerCase(Locale.ROOT)) {
            case "lightgbm" -> record.lightgbmPctError();
            case "gnn" -> record.gnnPctError();
            default -> record.pctError();
        };
    }

    private void putWaterFeatures(Map<String, Object> properties, PropertyRecord record) {
        WaterProximityService.WaterFeatures water = waterProximity.lookup(
            record.lat(), record.lng()
        );
        properties.put("distanceToWaterM", water.distanceToWaterM());
        properties.put(
            "waterProximity", WaterProximityService.waterProximity(water.distanceToWaterM())
        );
    }

    /** Add comparable nominal 90% and 95% intervals for all displayed models. */
    private void putPredictionIntervals(Map<String, Object> properties, PropertyRecord record) {
        double neural90 = 1.644854 * record.predictionStdPrice();
        double neural95 = 1.959964 * record.predictionStdPrice();
        properties.put("neuralLower90", Math.max(0.0, record.predictedPrice() - neural90));
        properties.put("neuralUpper90", record.predictedPrice() + neural90);
        properties.put("neuralLower95", Math.max(0.0, record.predictedPrice() - neural95));
        properties.put("neuralUpper95", record.predictedPrice() + neural95);
        putConformalInterval(properties, "lightgbm", record.lightgbmPredictedPrice(),
            lightgbmModel.getConformalLogResidual90(), lightgbmModel.getConformalLogResidual95());
        putConformalInterval(properties, "gnn", record.gnnPredictedPrice(),
            gnnModel.getConformalLogResidual90(), gnnModel.getConformalLogResidual95());
    }

    private static void putConformalInterval(
            Map<String, Object> properties, String model, double prediction,
            double logResidual90, double logResidual95) {
        properties.put(model + "Lower90", prediction * Math.exp(-logResidual90));
        properties.put(model + "Upper90", prediction * Math.exp(logResidual90));
        properties.put(model + "Lower95", prediction * Math.exp(-logResidual95));
        properties.put(model + "Upper95", prediction * Math.exp(logResidual95));
    }

    /** Annotate displayed RentCast records when the rolling snapshot post-dates their sale. */
    private void putRentcastPredictionStatus(Map<String, Object> properties, PropertyRecord record) {
        if (!"rentcast".equals(record.source())) return;
        boolean lookAhead = rentcastLocalMarket.hasLookAheadIssue(record.saleDate());
        properties.put("localMarketPredictionIssue",
            lookAhead ? "rolling_snapshot_look_ahead" : "");
        properties.put("rentcastLocalMarketLatestSaleDate",
            rentcastLocalMarket.latestSaleDate().toString());
    }

    /** Quarterly neural and LightGBM errors for every mapped community. */
    @GET
    @Path("/performance")
    public Map<String, Object> performance(
            @QueryParam("topN") @DefaultValue("10") int topN) {

        liveSources.ensureLoaded("sales");

        List<PropertyRecord> sales = store.getSalesRecords().stream()
            .filter(PropertyRecord::hasSalePrice)
            .filter(r -> r.community() != null && !r.community().isBlank())
            .collect(Collectors.toList());

        if (sales.isEmpty()) return Map.of(
            "communities", List.of(), "defaultCommunities", List.of(), "series", List.of()
        );

        // Find top N communities by count
        Map<String, Long> countByCommunity = sales.stream()
            .collect(Collectors.groupingBy(PropertyRecord::community, Collectors.counting()));
        List<String> allCommunities = countByCommunity.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
        List<String> topCommunities = allCommunities.stream().limit(topN).toList();
        List<Map<String, Object>> communityOptions = allCommunities.stream()
            .map(community -> Map.<String, Object>of(
                "id", community,
                "count", countByCommunity.getOrDefault(community, 0L)
            ))
            .toList();

        // Group by community + quarter
        // quarter key: "YYYY-QN"
        Map<String, Map<String, List<PropertyRecord>>> data = new LinkedHashMap<>();
        for (PropertyRecord r : sales) {
            if (r.saleDate() == null) continue;
            int year = r.saleDate().getYear();
            int q    = (r.saleDate().getMonthValue() - 1) / 3 + 1;
            String quarter = year + "-Q" + q;
            data.computeIfAbsent(r.community(), k -> new LinkedHashMap<>())
                .computeIfAbsent(quarter, k -> new ArrayList<>())
                .add(r);
        }

        // Build series per community
        List<Map<String, Object>> series = new ArrayList<>();
        for (String community : allCommunities) {
            Map<String, List<PropertyRecord>> byQuarter = data.getOrDefault(community, Map.of());
            List<Map<String, Object>> points = new ArrayList<>();
            for (Map.Entry<String, List<PropertyRecord>> e : new TreeMap<>(byQuarter).entrySet()) {
                List<PropertyRecord> records = e.getValue();
                double neuralMean = records.stream().mapToDouble(PropertyRecord::pctError).average().orElse(0);
                double lightgbmMean = records.stream().mapToDouble(PropertyRecord::lightgbmPctError).average().orElse(0);
                double gnnMean = records.stream().mapToDouble(PropertyRecord::gnnPctError).average().orElse(0);
                double neuralStd = Math.sqrt(records.stream()
                    .mapToDouble(r -> Math.pow(r.pctError() - neuralMean, 2)).average().orElse(0));
                double lightgbmStd = Math.sqrt(records.stream()
                    .mapToDouble(r -> Math.pow(r.lightgbmPctError() - lightgbmMean, 2)).average().orElse(0));
                Map<String, Object> point = new LinkedHashMap<>();
                point.put("quarter", e.getKey());
                point.put("neuralMean", neuralMean);
                point.put("neuralStd", neuralStd);
                point.put("lightgbmMean", lightgbmMean);
                point.put("lightgbmStd", lightgbmStd);
                point.put("gnnMean", gnnMean);
                point.put("gnnStd", Math.sqrt(records.stream().mapToDouble(r -> Math.pow(r.gnnPctError() - gnnMean, 2)).average().orElse(0)));
                point.put("neuralMape", records.stream()
                    .mapToDouble(r -> Math.abs(r.pctError())).average().orElse(0));
                point.put("lightgbmMape", records.stream()
                    .mapToDouble(r -> Math.abs(r.lightgbmPctError())).average().orElse(0));
                point.put("gnnMape", records.stream()
                    .mapToDouble(r -> Math.abs(r.gnnPctError())).average().orElse(0));
                point.put("actualMean", records.stream()
                    .mapToDouble(PropertyRecord::salePrice).average().orElse(0));
                point.put("neuralPredictedMean", records.stream()
                    .mapToDouble(PropertyRecord::predictedPrice).average().orElse(0));
                point.put("lightgbmPredictedMean", records.stream()
                    .mapToDouble(PropertyRecord::lightgbmPredictedPrice).average().orElse(0));
                point.put("gnnPredictedMean", records.stream()
                    .mapToDouble(PropertyRecord::gnnPredictedPrice).average().orElse(0));
                point.put("count", records.size());
                points.add(point);
            }
            series.add(Map.of(
                "community", community,
                "count",     countByCommunity.getOrDefault(community, 0L),
                "points",    points
            ));
        }

        return Map.of(
            "communities", communityOptions,
            "defaultCommunities", topCommunities,
            "series", series
        );
    }

    /**
     * Community layer: H3 L9 hexagons coloured by community ID for the map.
     */
    @GET
    @Path("/sales/community")
    public Map<String, Object> communityLayer() {
        liveSources.ensureLoaded("sales");
        List<PropertyRecord> sales = store.snapshot().completedSales();
        // Collect unique H3 L9 → community from sales records
        Map<String, String> h3ToCommunity = sales.stream()
            .filter(r -> r.h3Index() != null && r.community() != null && !r.community().isBlank())
            .collect(Collectors.toMap(
                PropertyRecord::h3Index,
                PropertyRecord::community,
                (a, b) -> a
            ));

        Map<String, double[]> communityStats = new HashMap<>();
        for (PropertyRecord r : sales) {
            if (r.h3Index() == null || r.community() == null || r.community().isBlank()) continue;
            double[] stats = communityStats.computeIfAbsent(r.community(), k -> new double[9]);
            stats[0] += 1;                       // count
            stats[1] += r.salePrice();
            stats[2] += r.predictedPrice();
            stats[3] += r.pctError();
            stats[4] += r.sqft();
            stats[5] += r.predictionStdPrice();
            stats[6] += r.lightgbmPredictedPrice();
            stats[7] += r.lightgbmPctError();
            stats[8] += r.sqftLot();
        }

        List<Map<String, Object>> features = new ArrayList<>();
        try {
            for (Map.Entry<String, String> entry : h3ToCommunity.entrySet()) {
                String hexId    = entry.getKey();
                String community = entry.getValue();
                List<double[]> coords = aggregation.boundaryFor(hexId);

                double[] stats = communityStats.getOrDefault(community, new double[9]);
                long salesCount = (long) stats[0];
                double meanSalePrice = salesCount > 0 ? stats[1] / salesCount : 0;
                double meanNeuralPredictedPrice = salesCount > 0 ? stats[2] / salesCount : 0;
                double avgNeuralPctError = salesCount > 0 ? stats[3] / salesCount : 0;
                double meanSqft = salesCount > 0 ? stats[4] / salesCount : 0;
                double meanPredStd = salesCount > 0 ? stats[5] / salesCount : 0;
                double meanLightgbmPredictedPrice = salesCount > 0 ? stats[6] / salesCount : 0;
                double avgLightgbmPctError = salesCount > 0 ? stats[7] / salesCount : 0;
                double meanSqftLot = salesCount > 0 ? stats[8] / salesCount : 0;

                Map<String, Object> properties = new LinkedHashMap<>();
                properties.put("community", community);
                properties.put("h3L9", hexId);
                properties.put("salesCount", salesCount);
                properties.put("meanSalePrice", meanSalePrice);
                properties.put("meanNeuralPredictedPrice", meanNeuralPredictedPrice);
                properties.put("meanLightgbmPredictedPrice", meanLightgbmPredictedPrice);
                properties.put("avgNeuralPctError", avgNeuralPctError);
                properties.put("avgLightgbmPctError", avgLightgbmPctError);
                properties.put("meanSqft", meanSqft);
                properties.put("meanSqftLot", meanSqftLot);
                properties.put("meanPredStd", meanPredStd);

                features.add(Map.of(
                    "type", "Feature",
                    "geometry", Map.of("type", "Polygon", "coordinates", List.of(coords)),
                    "properties", properties
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
        PropertyStore.Snapshot snapshot = store.snapshot();
        List<PropertyRecord> zillow = snapshot.zillow();
        List<PropertyRecord> sales = snapshot.completedSales();
        List<PropertyRecord> rentcast = snapshot.rentcast();

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("counts", Map.of(
            "sales", snapshot.sales().size(),
            "rentcast", snapshot.rentcast().size(),
            "zillow", snapshot.zillow().size()
        ));

        if (!zillow.isEmpty()) {
            double listTotal = 0;
            int listCount = 0;
            double neuralTotal = 0;
            double treeTotal = 0;
            for (PropertyRecord record : zillow) {
                if (record.hasSalePrice()) {
                    listTotal += record.salePrice();
                    listCount++;
                }
                neuralTotal += record.predictedPrice();
                treeTotal += record.lightgbmPredictedPrice();
            }
            result.put("zillow", Map.of(
                "total",        zillow.size(),
                "avgListPrice", listCount == 0 ? 0 : listTotal / listCount,
                "avgNeuralPredicted", neuralTotal / zillow.size(),
                "avgLightgbmPredicted", treeTotal / zillow.size()
            ));
        }

        if (!sales.isEmpty()) {
            ErrorStats values = errorStats(sales);
            result.put("sales", Map.of(
                "total",       sales.size(),
                "uniqueHexes", values.hexes.size(),
                "neuralAvgAbsError", values.avgNeuralError(),
                "lightgbmAvgAbsError", values.avgTreeError()
            ));
        }
            
        if (!rentcast.isEmpty()) {
            ErrorStats values = errorStats(rentcast);
            result.put("rentcast", Map.of(
                "total",       rentcast.size(),
                "uniqueSales", values.ids.size(),
                "uniqueHexes", values.hexes.size(),
                "neuralAvgAbsError", values.avgNeuralError(),
                "lightgbmAvgAbsError", values.avgTreeError()
            ));
        }

        return result;
    }

    private ErrorStats errorStats(List<PropertyRecord> records) {
        ErrorStats result = new ErrorStats();
        for (PropertyRecord record : records) {
            if (record.h3Index() != null) result.hexes.add(record.h3Index());
            if (record.id() != null) result.ids.add(record.id());
            if (record.hasSalePrice()) {
                result.count++;
                result.neuralError += Math.abs(record.pctError());
                result.treeError += Math.abs(record.lightgbmPctError());
            }
        }
        return result;
    }

    private static final class ErrorStats {
        int count;
        double neuralError;
        double treeError;
        final Set<String> hexes = new HashSet<>();
        final Set<String> ids = new HashSet<>();
        double avgNeuralError() { return count == 0 ? 0 : neuralError / count; }
        double avgTreeError() { return count == 0 ? 0 : treeError / count; }
    }
}
