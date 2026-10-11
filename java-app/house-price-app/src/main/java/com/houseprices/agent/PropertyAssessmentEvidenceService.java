package com.houseprices.agent;

import com.houseprices.ingest.PropertyRecord;
import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.FeatureContract;
import com.houseprices.model.GnnModel;
import com.houseprices.model.LightGBMModel;
import com.houseprices.model.ModelArtifacts;
import com.houseprices.model.PredictionContext;
import com.houseprices.model.PredictionContextFactory;
import com.houseprices.service.LiveSourceLoader;
import com.houseprices.service.PropertyStore;
import com.uber.h3core.H3Core;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.io.IOException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Builds the immutable, model-grounded evidence available to one agent run. */
@ApplicationScoped
public class PropertyAssessmentEvidenceService {

    private static final int MAX_NEARBY_LISTINGS = 12;
    private static final int MAX_CACHED_EVIDENCE = 128;

    @Inject PropertyStore store;
    @Inject LiveSourceLoader liveSources;
    @Inject ModelArtifacts artifacts;
    @Inject FeatureContract featureContract;
    @Inject PredictionContextFactory contextFactory;
    @Inject EmbeddingModel neuralModel;
    @Inject LightGBMModel lightgbmModel;
    @Inject GnnModel gnnModel;
    @Inject H3MarketSurfaceService marketSurface;

    private final H3Core h3;
    private final Map<String, CachedEvidence> evidenceCache = Collections.synchronizedMap(
        new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, CachedEvidence> eldest) {
                return size() > MAX_CACHED_EVIDENCE;
            }
        }
    );

    public PropertyAssessmentEvidenceService() {
        try { h3 = H3Core.newInstance(); }
        catch (IOException exception) { throw new RuntimeException("H3 initialization failed", exception); }
    }

    public EvidenceBundle build(String listingId) {
        liveSources.ensureLoaded("redfin");
        PropertyRecord listing = store.findBySourceAndId("redfin", listingId)
            .orElseThrow(() -> new IllegalArgumentException("Redfin listing not found: " + listingId));
        long storeVersion = store.version();
        CachedEvidence cached = evidenceCache.get(listingId);
        if (cached != null && cached.listing() == listing && cached.storeVersion() == storeVersion) {
            return cached.evidence();
        }
        LocalDate date = listing.saleDate() == null ? LocalDate.now() : listing.saleDate();
        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            listing.h3Index(), date, listing.sqft(), listing.sqftLot(), listing.beds(),
            listing.lat(), listing.lng(), FeatureContract.DEFAULT_MORTGAGE_RATE,
            FeatureContract.DEFAULT_UNEMPLOYMENT_RATE
        );
        LocalDate latestSnapshotSale = artifacts.getLocalMarketLatestSaleDate();
        boolean demonstration = latestSnapshotSale != null && !date.isAfter(latestSnapshotSale);
        PredictionContext context = contextFactory.prepare(input, demonstration);
        EvidenceBundle evidence = new EvidenceBundle(
            listingEvidence(listing), predictionEvidence(listing, context, demonstration),
            h3ModelArtifactEvidence(listing, context)
        );
        evidenceCache.put(listingId, new CachedEvidence(listing, storeVersion, evidence));
        return evidence;
    }

    private Map<String, Object> listingEvidence(PropertyRecord value) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", value.id());
        result.put("address", value.address());
        result.put("listingUrl", value.listingUrl());
        result.put("observationDate", value.saleDate());
        result.put("listPrice", value.salePrice());
        result.put("latitude", value.lat());
        result.put("longitude", value.lng());
        result.put("h3Index", value.h3Index());
        result.put("community", value.community());
        result.put("sqft", value.sqft());
        result.put("lotSqft", value.sqftLot());
        result.put("beds", value.beds());
        result.put("baths", value.baths());
        result.put("homeType", value.homeType());
        result.put("description", value.description());
        result.put("atAGlanceFacts", value.atAGlanceFacts());
        result.put("h3LocationAndHistoricalMarket", marketSurface.lookup(value.h3Index()));
        return result;
    }

    private Map<String, Object> predictionEvidence(
            PropertyRecord listing, PredictionContext context, boolean demonstration) {
        EmbeddingModel.ShapleyExplanation neuralExplanation = neuralModel.explainPrepared(context);
        LightGBMModel.PredictionExplanation lightgbmExplanation = lightgbmModel.explainPrepared(context);
        GnnModel.ShapleyExplanation gnnExplanation = gnnModel.explain(new GnnModel.Input(
            listing.h3Index(), context.input().saleDate(), listing.sqft(), listing.sqftLot(), listing.beds(),
            listing.lat(), listing.lng()
        ));
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("predictionDate", context.input().saleDate());
        result.put("modelSnapshotReconstruction", demonstration);
        result.put("predictions", predictionSummary(listing));
        result.put("rawModelInputs", rawInputs(context));
        result.put("lightgbmTreeShap", lightgbmExplanation);
        result.put("neuralShapley", neuralExplanation);
        result.put("spatialGnnShapley", gnnExplanation);
        result.put("timeEvidence", Map.of(
            "observationDate", context.input().saleDate(),
            "rawTimeInputs", named(featureContract.timeFeatures(), context.rawTime()),
            "rawMarketInputs", named(featureContract.marketFeatures(), context.rawMarket()),
            "method", "Uses the stored listing predictions and their deployed-model explanations; no monthly counterfactual series was generated."
        ));
        return result;
    }

    private Map<String, Object> predictionSummary(PropertyRecord listing) {
        double[] values = {
            listing.predictedPrice(), listing.lightgbmPredictedPrice(), listing.gnnPredictedPrice()
        };
        double[] sorted = values.clone();
        java.util.Arrays.sort(sorted);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("listPrice", listing.salePrice());
        result.put("neural", listing.predictedPrice());
        result.put("lightgbm", listing.lightgbmPredictedPrice());
        result.put("spatialGnn", listing.gnnPredictedPrice());
        result.put("modelMedian", sorted[1]);
        result.put("neuralVsListPct", listing.pctError());
        result.put("lightgbmVsListPct", listing.lightgbmPctError());
        result.put("spatialGnnVsListPct", listing.gnnPctError());
        result.put("neuralStdDollars", listing.predictionStdPrice());
        result.put("neural95Lower", Math.max(0, listing.predictedPrice() - 1.96 * listing.predictionStdPrice()));
        result.put("neural95Upper", listing.predictedPrice() + 1.96 * listing.predictionStdPrice());
        result.put("neural95WidthPct", listing.predictionCvPct());
        if (listing.clsAttention() != null && listing.clsAttention().length == 4) {
            result.put("neuralAttention", Map.of(
                "community", listing.clsAttention()[0], "property", listing.clsAttention()[1],
                "time", listing.clsAttention()[2], "market", listing.clsAttention()[3]
            ));
        }
        return result;
    }

    private Map<String, Object> rawInputs(PredictionContext context) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("property", named(featureContract.propertyFeatures(), context.rawProperty()));
        result.put("time", named(featureContract.timeFeatures(), context.rawTime()));
        result.put("market", named(featureContract.marketFeatures(), context.rawMarket()));
        result.put("h3LocalMarketRows", localMarketRows(context));
        return result;
    }

    private List<Map<String, Object>> localMarketRows(PredictionContext context) {
        String[] cells = artifacts.lookupH3NeighborCells(context.input().h3Index());
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int index = 0; index < context.rawLocalMarket().length; index++) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("relation", index == 0 ? "selected cell" : "model neighbor " + index);
            row.put("h3Index", cells[index]);
            row.put("communityIndex", context.communities()[index]);
            row.put("features", named(
                featureContract.localMarketFeatures(), context.rawLocalMarket()[index]
            ));
            rows.add(row);
        }
        return rows;
    }

    private Map<String, Double> named(List<String> names, double[] values) {
        Map<String, Double> result = new LinkedHashMap<>();
        for (int index = 0; index < Math.min(names.size(), values.length); index++) {
            result.put(names.get(index), values[index]);
        }
        return result;
    }

    private Map<String, Double> named(List<String> names, float[] values) {
        Map<String, Double> result = new LinkedHashMap<>();
        for (int index = 0; index < Math.min(names.size(), values.length); index++) {
            result.put(names.get(index), (double) values[index]);
        }
        return result;
    }

    private Map<String, Object> h3ModelArtifactEvidence(
            PropertyRecord listing, PredictionContext context) {
        long origin = h3.stringToH3(listing.h3Index());
        Set<String> ringTwo = new LinkedHashSet<>();
        for (long cell : h3.gridDisk(origin, 2)) ringTwo.add(h3.h3ToString(cell));
        List<Map<String, Object>> nearbyListings = store.getRedfinListings().stream()
            .filter(value -> !value.id().equals(listing.id()))
            .filter(value -> ringTwo.contains(value.h3Index()))
            .sorted(Comparator.comparingDouble(value -> nearbyListingScore(listing, value, origin)))
            .limit(MAX_NEARBY_LISTINGS)
            .map(value -> nearbyListing(listing, value, origin))
            .toList();
        String[] alignedCells = artifacts.lookupH3NeighborCells(listing.h3Index());
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("selectedH3", listing.h3Index());
        result.put("relationship", "Deployed-model center cell plus six aligned H3 neighbor rows; nearby Redfin listings are limited to H3 rings 0–2.");
        result.put("artifactLatestSaleDate", artifacts.getLocalMarketLatestSaleDate());
        result.put("modelArtifactLocalMarketRows", localMarketRows(context));
        result.put("historicalMarketSurface", marketSurface.lookupAlignedNeighborhood(alignedCells));
        result.put("historicalMarketSurfaceMetadata", marketSurface.metadata());
        result.put("nearbyRedfinListingsWithStoredPredictions", nearbyListings);
        result.put("historicalSalesCsvUsedAtRuntime", false);
        return result;
    }

    private Map<String, Object> nearbyListing(
            PropertyRecord selected, PropertyRecord nearby, long origin) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", nearby.id());
        result.put("address", nearby.address());
        result.put("h3Index", nearby.h3Index());
        result.put("ringDistance", gridDistance(origin, nearby.h3Index()));
        result.put("distanceKm", haversineKm(selected.lat(), selected.lng(), nearby.lat(), nearby.lng()));
        result.put("observationDate", nearby.saleDate());
        result.put("listPrice", nearby.salePrice());
        result.put("sqft", nearby.sqft());
        result.put("beds", nearby.beds());
        result.put("baths", nearby.baths());
        result.put("homeType", nearby.homeType());
        result.put("storedNeuralPrediction", nearby.predictedPrice());
        result.put("storedLightgbmPrediction", nearby.lightgbmPredictedPrice());
        result.put("storedSpatialGnnPrediction", nearby.gnnPredictedPrice());
        return result;
    }

    private double nearbyListingScore(PropertyRecord selected, PropertyRecord nearby, long origin) {
        double sqftDelta = selected.sqft() > 0
            ? Math.abs(nearby.sqft() - selected.sqft()) / selected.sqft() : 1;
        return gridDistance(origin, nearby.h3Index()) * 2.0 + sqftDelta * 3.0
            + Math.abs(nearby.beds() - selected.beds()) * 0.5;
    }

    private int gridDistance(long origin, String cell) {
        try { return (int) h3.gridDistance(origin, h3.stringToH3(cell)); }
        catch (Exception ignored) { return 99; }
    }

    private double haversineKm(double lat1, double lng1, double lat2, double lng2) {
        double latDelta = Math.toRadians(lat2 - lat1);
        double lngDelta = Math.toRadians(lng2 - lng1);
        double a = Math.sin(latDelta / 2) * Math.sin(latDelta / 2)
            + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
            * Math.sin(lngDelta / 2) * Math.sin(lngDelta / 2);
        return 6371.0088 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    public record EvidenceBundle(
        Map<String, Object> listing,
        Map<String, Object> prediction,
        Map<String, Object> h3Market
    ) {
        public Map<String, Object> snapshot() {
            return Map.of("listing", listing, "prediction", prediction, "h3Market", h3Market);
        }
    }

    private record CachedEvidence(
        PropertyRecord listing, long storeVersion, EvidenceBundle evidence
    ) {}
}
