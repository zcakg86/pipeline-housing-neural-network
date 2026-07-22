package com.houseprices.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.LightGBMModel;
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
        double conformalLogResidual = lightgbmModel.getConformalLogResidual95();
        double lightgbmLowerFactor = Math.exp(-conformalLogResidual);
        double lightgbmUpperFactor = Math.exp(conformalLogResidual);

        List<Map<String, Object>> features = new ArrayList<>(cells.size());
        for (int i = 0; i < cells.size(); i++) {
            GridCell cell = cells.get(i);
            EmbeddingModel.PredictionResult neuralPrediction = neural[i];
            float[] attention = neuralPrediction.clsAttention();
            double neuralMargin = 1.96 * neuralPrediction.predictionStdPrice();

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
            properties.put("isWaterfront", water.isWaterfront() == 1.0);
            properties.put("neuralPredictedPrice", neuralPrediction.predictedPrice());
            properties.put(
                "neuralLower95", Math.max(0.0, neuralPrediction.predictedPrice() - neuralMargin)
            );
            properties.put("neuralUpper95", neuralPrediction.predictedPrice() + neuralMargin);
            properties.put("lightgbmPredictedPrice", lightgbm[i]);
            properties.put("lightgbmLower95", lightgbm[i] * lightgbmLowerFactor);
            properties.put("lightgbmUpper95", lightgbm[i] * lightgbmUpperFactor);
            properties.put("attnCommunity", attention[0]);
            properties.put("attnYear", attention[1]);
            properties.put("attnWeek", attention[2]);
            properties.put("attnProperty", attention[3]);
            properties.put("attnTime", attention[4]);
            properties.put("attnMarket", attention[5]);

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

    /** Exact LightGBM plus exact-group and sampled-feature neural contributions. */
    public Map<String, Object> explain(String h3Index, LocalDate saleDate) {
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
        MarketIndicatorService.MarketIndicators market =
            marketIndicators.lookup(predictionDate);
        EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
            cell.h3L8(), predictionDate, SQFT, SQFT_LOT, BEDS,
            cell.lat(), cell.lng(),
            market.mortgageRate(), market.unemploymentRate()
        );
        PredictionContext context = contextFactory.prepare(input, true);
        LightGBMModel.PredictionExplanation explanation =
            lightgbmModel.explainPrepared(context);
        EmbeddingModel.ShapleyExplanation neuralExplanation =
            neuralModel.explainPrepared(context);
        return Map.of(
            "h3Index", h3Index,
            "saleDate", predictionDate.toString(),
            "marketIndicatorDate", market.effectiveDate().toString(),
            "mortgageRate", market.mortgageRate(),
            "unemploymentRate", market.unemploymentRate(),
            "explanation", explanation,
            "neuralExplanation", neuralExplanation
        );
    }

    public LocalDate minimumPredictionDate() {
        return MINIMUM_DEMO_DATE;
    }

    public int size() {
        return cells.size();
    }
}
