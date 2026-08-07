package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;
import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.LightGBMModel;
import com.houseprices.model.GnnModel;
import com.houseprices.model.MarketIndicatorService;
import com.houseprices.model.ModelArtifacts;
import com.houseprices.model.PredictionContext;
import com.houseprices.model.PredictionContextFactory;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/** Builds an on-demand monthly valuation history for one stored property. */
@ApplicationScoped
public class PropertyTimeSeriesService {

    @Inject PropertyStore store;
    @Inject ModelArtifacts artifacts;
    @Inject MarketIndicatorService marketIndicators;
    @Inject PredictionContextFactory contextFactory;
    @Inject EmbeddingModel neuralModel;
    @Inject LightGBMModel lightgbmModel;
    @Inject GnnModel gnnModel;

    public Map<String, Object> predict(String source, String id) {
        PropertyRecord property = store.findBySourceAndId(source, id)
            .orElseThrow(() -> new IllegalArgumentException("Property record not found"));
        LocalDate highlightDate = property.saleDate() == null
            ? LocalDate.now() : property.saleDate();
        LocalDate endDate = highlightDate.isAfter(LocalDate.now())
            ? highlightDate : LocalDate.now();

        // Use a stable month cadence from the model reference date and insert
        // the observation's exact date so its original prediction is visible.
        TreeSet<LocalDate> dates = new TreeSet<>();
        LocalDate cursor = artifacts.getReferenceDate();
        while (!cursor.isAfter(endDate)) {
            dates.add(cursor);
            cursor = cursor.plusMonths(1);
        }
        dates.add(highlightDate);
        dates.add(endDate);

        LocalDate latestSnapshotSale = artifacts.getLocalMarketLatestSaleDate();
        List<PredictionContext> contexts = new ArrayList<>(dates.size());
        List<MarketIndicatorService.MarketIndicators> indicators =
            new ArrayList<>(dates.size());
        List<Boolean> demonstrations = new ArrayList<>(dates.size());
        for (LocalDate date : dates) {
            MarketIndicatorService.MarketIndicators market = marketIndicators.lookup(date);
            indicators.add(market);
            EmbeddingModel.BatchInput input = new EmbeddingModel.BatchInput(
                property.h3Index(), date, property.sqft(), property.sqftLot(),
                property.beds(), property.lat(), property.lng(),
                market.mortgageRate(), market.unemploymentRate()
            );
            boolean demonstration = latestSnapshotSale != null
                && !date.isAfter(latestSnapshotSale);
            demonstrations.add(demonstration);
            contexts.add(contextFactory.prepare(input, demonstration));
        }

        EmbeddingModel.PredictionResult[] neural = neuralModel.predictPrepared(contexts);
        double[] lightgbm = lightgbmModel.predictPrepared(contexts);
        double[] gnn = gnnModel.predictBatch(dates.stream().map(date -> new GnnModel.Input(
            property.h3Index(), date, property.sqft(), property.sqftLot(), property.beds(),
            property.lat(), property.lng()
        )).toList());
        List<Map<String, Object>> points = new ArrayList<>(dates.size());
        int index = 0;
        for (LocalDate date : dates) {
            Map<String, Object> point = new LinkedHashMap<>();
            point.put("date", date.toString());
            point.put("neuralPredictedPrice", neural[index].predictedPrice());
            point.put("lightgbmPredictedPrice", lightgbm[index]);
            point.put("gnnPredictedPrice", gnn[index]);
            point.put("marketIndicatorDate", indicators.get(index).effectiveDate().toString());
            point.put("historicalSnapshotReconstruction", demonstrations.get(index));
            point.put("highlight", date.equals(highlightDate));
            points.add(point);
            index++;
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("source", source);
        result.put("id", id);
        result.put("address", property.address());
        result.put("highlightDate", highlightDate.toString());
        result.put("highlightLabel", "sales".equals(source) || "rentcast".equals(source)
            ? "Sale date" : "Initial prediction date");
        result.put("actualPrice", property.salePrice());
        result.put("storedNeuralPrediction", property.predictedPrice());
        result.put("storedLightgbmPrediction", property.lightgbmPredictedPrice());
        result.put("storedGnnPrediction", property.gnnPredictedPrice());
        result.put("points", points);
        return result;
    }
}
