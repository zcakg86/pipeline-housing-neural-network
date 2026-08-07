package com.houseprices.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.LightGBMModel;
import com.houseprices.model.GnnModel;
import com.houseprices.model.MarketIndicatorService;
import com.houseprices.model.ModelArtifacts;
import com.houseprices.model.PredictionContext;
import com.houseprices.model.PredictionContextFactory;
import com.houseprices.model.WaterProximityService;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Produces date-sensitive predictions for the precomputed H3 level-8 grid. */
@ApplicationScoped
public class SyntheticPredictionService {

    private static final Logger LOG = Logger.getLogger(SyntheticPredictionService.class);
    private static final double SQFT = 2000.0;
    private static final double SQFT_LOT = 4000.0;
    private static final double BEDS = 3.0;
    private static final int CACHE_SIZE = 3;
    private static final LocalDate MINIMUM_DEMO_DATE = LocalDate.of(2020, 1, 1);

    @Inject EmbeddingModel neuralModel;
    @Inject LightGBMModel lightgbmModel;
    @Inject GnnModel gnnModel;
    @Inject ModelArtifacts artifacts;
    @Inject WaterProximityService waterProximity;
    @Inject MarketIndicatorService marketIndicators;
    @Inject PredictionContextFactory contextFactory;

    private List<GridCell> cells;
    private Map<String, GridCell> cellsByH3;
    private final Map<LocalDate, Map<String, Object>> cache =
        new LinkedHashMap<>(4, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<LocalDate, Map<String, Object>> eldest) {
                return size() > CACHE_SIZE;
            }
        };

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record GridFile(
        @JsonProperty("cell_count") int cellCount,
        List<GridCell> cells
    ) {}

    public record GridCell(
        @JsonProperty("h3_l8") String h3L8,
        double lat,
        double lng,
        List<List<Double>> boundary
    ) {}

    @PostConstruct
    void loadGrid() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(
                "model-artifacts/synthetic_h3_l8_grid.json")) {
            if (input == null) {
                throw new IllegalStateException("synthetic_h3_l8_grid.json not found in resources");
            }
            GridFile grid = new ObjectMapper().readValue(input, GridFile.class);
            if (grid.cells() == null || grid.cells().size() != grid.cellCount()) {
                throw new IllegalStateException("Synthetic H3 grid count does not match its metadata");
            }
            cells = List.copyOf(grid.cells());
            Map<String, GridCell> indexedCells = new LinkedHashMap<>();
            for (GridCell cell : cells) indexedCells.put(cell.h3L8(), cell);
            cellsByH3 = Map.copyOf(indexedCells);
            LOG.infof("Loaded %,d synthetic H3 level-8 cells", cells.size());
        } catch (Exception e) {
            throw new RuntimeException("Failed to load synthetic H3 grid", e);
        }
    }

    public synchronized Map<String, Object> predict(LocalDate saleDate) {
        LocalDate predictionDate = saleDate == null ? LocalDate.now() : saleDate;
        LocalDate minimumDate = minimumPredictionDate();
        if (predictionDate.isBefore(minimumDate)) {
            throw new IllegalArgumentException(
                "Synthetic prediction date " + predictionDate +
                " must be on or after " + minimumDate
            );
        }
        Map<String, Object> cached = cache.get(predictionDate);
        if (cached != null) return cached;

        long started = System.nanoTime();
        MarketIndicatorService.MarketIndicators market =
            marketIndicators.lookup(predictionDate);
        List<EmbeddingModel.BatchInput> inputs = cells.stream()
            .map(cell -> new EmbeddingModel.BatchInput(
                cell.h3L8(), predictionDate, SQFT, SQFT_LOT, BEDS,
                cell.lat(), cell.lng(),
                market.mortgageRate(), market.unemploymentRate()
            ))
            .toList();
        List<PredictionContext> contexts = contextFactory.prepareAll(inputs, true);
        EmbeddingModel.PredictionResult[] neural = neuralModel.predictPrepared(contexts);
        double[] lightgbm = lightgbmModel.predictPrepared(contexts);
        double[] gnn = gnnModel.predictBatch(inputs.stream().map(input -> new GnnModel.Input(
            input.h3Index(), input.saleDate(), input.sqft(), input.sqftLot(), input.beds(),
            input.latitude(), input.longitude()
        )).toList());
        double conformalLogResidual = lightgbmModel.getConformalLogResidual95();
        double lightgbmLogResidual90 = lightgbmModel.getConformalLogResidual90();
        double lightgbmLowerFactor = Math.exp(-conformalLogResidual);
        double lightgbmUpperFactor = Math.exp(conformalLogResidual);
        double lightgbmLower90Factor = Math.exp(-lightgbmLogResidual90);
        double lightgbmUpper90Factor = Math.exp(lightgbmLogResidual90);
        double gnnLogResidual90 = gnnModel.getConformalLogResidual90();
        double gnnLogResidual95 = gnnModel.getConformalLogResidual95();
        double gnnLower90Factor = Math.exp(-gnnLogResidual90);
        double gnnUpper90Factor = Math.exp(gnnLogResidual90);
        double gnnLower95Factor = Math.exp(-gnnLogResidual95);
        double gnnUpper95Factor = Math.exp(gnnLogResidual95);

        List<Map<String, Object>> features = new ArrayList<>(cells.size());
        for (int i = 0; i < cells.size(); i++) {
            GridCell cell = cells.get(i);
            EmbeddingModel.PredictionResult neuralPrediction = neural[i];
            float[] attention = neuralPrediction.clsAttention();
            double neuralMargin = 1.96 * neuralPrediction.predictionStdPrice();
            double neuralMargin90 = 1.644854 * neuralPrediction.predictionStdPrice();

            Map<String, Object> properties = new LinkedHashMap<>();
            properties.put("h3Index", cell.h3L8());
            properties.put("community", artifacts.lookupCommunity(cell.h3L8()));
            properties.put("lat", cell.lat());
            properties.put("lng", cell.lng());
            properties.put("saleDate", predictionDate.toString());
            properties.put("sqft", SQFT);
            properties.put("sqftLot", SQFT_LOT);
            properties.put("beds", BEDS);
            WaterProximityService.WaterFeatures water = waterProximity.lookup(
                cell.lat(), cell.lng()
            );
            properties.put("distanceToWaterM", water.distanceToWaterM());
            properties.put(
                "waterProximity", WaterProximityService.waterProximity(water.distanceToWaterM())
            );
            properties.put("neuralPredictedPrice", neuralPrediction.predictedPrice());
            properties.put(
                "neuralLower95", Math.max(0.0, neuralPrediction.predictedPrice() - neuralMargin)
            );
            properties.put("neuralUpper95", neuralPrediction.predictedPrice() + neuralMargin);
            properties.put("neuralLower90", Math.max(0.0, neuralPrediction.predictedPrice() - neuralMargin90));
            properties.put("neuralUpper90", neuralPrediction.predictedPrice() + neuralMargin90);
            properties.put("lightgbmPredictedPrice", lightgbm[i]);
            properties.put("gnnPredictedPrice", gnn[i]);
            properties.put("lightgbmLower90", lightgbm[i] * lightgbmLower90Factor);
            properties.put("lightgbmUpper90", lightgbm[i] * lightgbmUpper90Factor);
            properties.put("lightgbmLower95", lightgbm[i] * lightgbmLowerFactor);
            properties.put("lightgbmUpper95", lightgbm[i] * lightgbmUpperFactor);
            properties.put("gnnLower90", gnn[i] * gnnLower90Factor);
            properties.put("gnnUpper90", gnn[i] * gnnUpper90Factor);
            properties.put("gnnLower95", gnn[i] * gnnLower95Factor);
            properties.put("gnnUpper95", gnn[i] * gnnUpper95Factor);
            properties.put("attnCommunity", attention[0]);
            properties.put("attnProperty", attention[1]);
            properties.put("attnTime", attention[2]);
            properties.put("attnMarket", attention[3]);

            features.add(Map.of(
                "type", "Feature",
                "geometry", Map.of(
                    "type", "Polygon",
                    "coordinates", List.of(cell.boundary())
                ),
                "properties", properties
            ));
        }

        Map<String, Object> result = Map.of(
            "type", "FeatureCollection",
            "saleDate", predictionDate.toString(),
            "minimumSaleDate", minimumDate.toString(),
            "marketIndicatorDate", market.effectiveDate().toString(),
            "mortgageRate", market.mortgageRate(),
            "unemploymentRate", market.unemploymentRate(),
            "count", features.size(),
            "features", features
        );
        cache.put(predictionDate, result);
        LOG.infof(
            "Predicted %,d synthetic properties for %s in %.2f seconds",
            cells.size(), predictionDate, (System.nanoTime() - started) / 1_000_000_000.0
        );
        return result;
    }

    /** Exact LightGBM, neural, and deployable GNN group contributions. */
    public Map<String, Object> explain(String h3Index, LocalDate saleDate) {
        return explain(h3Index, saleDate, null, null);
    }

    /** Contributions at either the cell centroid or a user-adjusted coordinate. */
    public Map<String, Object> explain(
            String h3Index, LocalDate saleDate, Double latitude, Double longitude) {
        LocalDate predictionDate = saleDate == null ? LocalDate.now() : saleDate;
        if (predictionDate.isBefore(minimumPredictionDate())) {
            throw new IllegalArgumentException(
                "Synthetic prediction date " + predictionDate +
                " must be on or after " + minimumPredictionDate()
            );
        }
        GridCell cell = cellsByH3.get(h3Index);
        if (cell == null) {
            throw new IllegalArgumentException("Unknown synthetic H3 cell: " + h3Index);
        }
        double predictionLat = validCoordinate(latitude, -90.0, 90.0)
            ? latitude : cell.lat();
        double predictionLng = validCoordinate(longitude, -180.0, 180.0)
            ? longitude : cell.lng();
        MarketIndicatorService.MarketIndicators market =
            marketIndicators.lookup(predictionDate);
        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            cell.h3L8(), predictionDate, SQFT, SQFT_LOT, BEDS,
            predictionLat, predictionLng,
            market.mortgageRate(), market.unemploymentRate()
        );
        PredictionContext context = contextFactory.prepare(input, true);
        LightGBMModel.PredictionExplanation explanation =
            lightgbmModel.explainPrepared(context);
        EmbeddingModel.ShapleyExplanation neuralExplanation =
            neuralModel.explainPrepared(context);
        GnnModel.ShapleyExplanation gnnExplanation = gnnModel.explain(new GnnModel.Input(
            cell.h3L8(), predictionDate, SQFT, SQFT_LOT, BEDS,
            predictionLat, predictionLng
        ));
        return Map.of(
            "h3Index", h3Index,
            "lat", predictionLat,
            "lng", predictionLng,
            "saleDate", predictionDate.toString(),
            "marketIndicatorDate", market.effectiveDate().toString(),
            "mortgageRate", market.mortgageRate(),
            "unemploymentRate", market.unemploymentRate(),
            "explanation", explanation,
            "neuralExplanation", neuralExplanation,
            "gnnExplanation", gnnExplanation
        );
    }

    /** Recalculate one synthetic observation after its map marker is moved. */
    public Map<String, Object> predictPoint(
            String h3Index, LocalDate saleDate, Double latitude, Double longitude) {
        LocalDate predictionDate = saleDate == null ? LocalDate.now() : saleDate;
        if (predictionDate.isBefore(minimumPredictionDate())) {
            throw new IllegalArgumentException(
                "Synthetic prediction date " + predictionDate +
                " must be on or after " + minimumPredictionDate()
            );
        }
        GridCell cell = cellsByH3.get(h3Index);
        if (cell == null) {
            throw new IllegalArgumentException("Unknown synthetic H3 cell: " + h3Index);
        }
        if (!validCoordinate(latitude, -90.0, 90.0)
                || !validCoordinate(longitude, -180.0, 180.0)) {
            throw new IllegalArgumentException("Valid latitude and longitude are required");
        }
        MarketIndicatorService.MarketIndicators market =
            marketIndicators.lookup(predictionDate);
        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            cell.h3L8(), predictionDate, SQFT, SQFT_LOT, BEDS,
            latitude, longitude, market.mortgageRate(), market.unemploymentRate()
        );
        PredictionContext context = contextFactory.prepare(input, true);
        EmbeddingModel.PredictionResult neural =
            neuralModel.predictPrepared(List.of(context))[0];
        double lightgbm = lightgbmModel.predictPrepared(List.of(context))[0];
        double gnn = gnnModel.predict(
            cell.h3L8(), predictionDate, SQFT, SQFT_LOT, BEDS, latitude, longitude
        );
        double neuralMargin = 1.96 * neural.predictionStdPrice();
        double neuralMargin90 = 1.644854 * neural.predictionStdPrice();
        double lightgbmMargin = lightgbmModel.getConformalLogResidual95();
        double lightgbmMargin90 = lightgbmModel.getConformalLogResidual90();
        double gnnMargin90 = gnnModel.getConformalLogResidual90();
        double gnnMargin95 = gnnModel.getConformalLogResidual95();
        WaterProximityService.WaterFeatures water =
            waterProximity.lookup(latitude, longitude);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("h3Index", h3Index);
        result.put("community", artifacts.lookupCommunity(h3Index));
        result.put("lat", latitude);
        result.put("lng", longitude);
        result.put("centerLat", cell.lat());
        result.put("centerLng", cell.lng());
        result.put("saleDate", predictionDate.toString());
        result.put("sqft", SQFT);
        result.put("sqftLot", SQFT_LOT);
        result.put("beds", BEDS);
        result.put("distanceToWaterM", water.distanceToWaterM());
        result.put(
            "waterProximity", WaterProximityService.waterProximity(water.distanceToWaterM())
        );
        result.put("neuralPredictedPrice", neural.predictedPrice());
        result.put("neuralLower95", Math.max(0.0, neural.predictedPrice() - neuralMargin));
        result.put("neuralUpper95", neural.predictedPrice() + neuralMargin);
        result.put("neuralLower90", Math.max(0.0, neural.predictedPrice() - neuralMargin90));
        result.put("neuralUpper90", neural.predictedPrice() + neuralMargin90);
        result.put("lightgbmPredictedPrice", lightgbm);
        result.put("gnnPredictedPrice", gnn);
        result.put("lightgbmLower90", lightgbm * Math.exp(-lightgbmMargin90));
        result.put("lightgbmUpper90", lightgbm * Math.exp(lightgbmMargin90));
        result.put("lightgbmLower95", lightgbm * Math.exp(-lightgbmMargin));
        result.put("lightgbmUpper95", lightgbm * Math.exp(lightgbmMargin));
        result.put("gnnLower90", gnn * Math.exp(-gnnMargin90));
        result.put("gnnUpper90", gnn * Math.exp(gnnMargin90));
        result.put("gnnLower95", gnn * Math.exp(-gnnMargin95));
        result.put("gnnUpper95", gnn * Math.exp(gnnMargin95));
        result.put("attnCommunity", neural.clsAttention()[0]);
        result.put("attnProperty", neural.clsAttention()[1]);
        result.put("attnTime", neural.clsAttention()[2]);
        result.put("attnMarket", neural.clsAttention()[3]);
        result.put("marketIndicatorDate", market.effectiveDate().toString());
        result.put("mortgageRate", market.mortgageRate());
        result.put("unemploymentRate", market.unemploymentRate());
        return result;
    }

    private static boolean validCoordinate(Double value, double minimum, double maximum) {
        return value != null && Double.isFinite(value)
            && value >= minimum && value <= maximum;
    }

    public LocalDate minimumPredictionDate() {
        return MINIMUM_DEMO_DATE;
    }

    public int size() {
        return cells.size();
    }
}
