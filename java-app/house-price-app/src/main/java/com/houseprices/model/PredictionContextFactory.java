package com.houseprices.model;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.time.temporal.ChronoUnit;
import java.time.temporal.IsoFields;
import java.util.ArrayList;
import java.util.List;

/** Prepares date, spatial, water, market and local inputs exactly once. */
@ApplicationScoped
public class PredictionContextFactory {

    @Inject ModelArtifacts artifacts;
    @Inject WaterProximityService waterProximity;
    @Inject FeatureContract featureContract;

    public PredictionContext prepare(EmbeddingModel.BatchInput input, boolean demonstrationMode) {
        if (input.saleDate() == null) {
            throw new IllegalArgumentException("Prediction date is required");
        }
        int[] communities = artifacts.lookupH3Neighbors(input.h3Index());
        long year = artifacts.lookupYear(input.saleDate().getYear());
        long week = artifacts.lookupWeek(
            input.saleDate().get(IsoFields.WEEK_OF_WEEK_BASED_YEAR)
        );
        WaterProximityService.WaterFeatures water = water(input);
        List<String> propertyNames = featureContract.propertyFeatures();
        double[] rawProperty = new double[propertyNames.size()];
        float[] scaledProperty = new float[propertyNames.size()];
        for (int index = 0; index < propertyNames.size(); index++) {
            String name = propertyNames.get(index);
            double value = switch (name) {
                case "sqft" -> input.sqft();
                case "sqft_lot" -> input.sqftLot();
                case "beds" -> input.beds();
                case "water_proximity" -> WaterProximityService.waterProximity(
                    water.distanceToWaterM()
                );
                case "distance_to_water_m" -> water.distanceToWaterM();
                case "is_waterfront" -> water.isWaterfront();
                default -> throw new IllegalStateException("Unsupported property feature: " + name);
            };
            rawProperty[index] = value;
            scaledProperty[index] = (float) artifacts.scaleFeature(name, value);
        }
        double rawTime = ChronoUnit.DAYS.between(
            artifacts.getReferenceDate(), input.saleDate()
        ) / 365.25;
        float[] rawMarket = {
            (float) input.mortgageRate(), (float) input.unemploymentRate()
        };
        float[] scaledMarket = {
            (float) artifacts.scaleFeature("mortgage_rate", input.mortgageRate()),
            (float) artifacts.scaleFeature("unemployment_rate", input.unemploymentRate())
        };
        float[][] rawLocal = demonstrationMode
            ? artifacts.lookupDemoRawLocalMarketFeatures(input.h3Index(), input.saleDate())
            : artifacts.lookupRawLocalMarketFeatures(input.h3Index(), input.saleDate());
        List<String> localFeatures = featureContract.localMarketFeatures();
        float[][] scaledLocal = new float[rawLocal.length][localFeatures.size()];
        for (int ring = 0; ring < rawLocal.length; ring++) {
            for (int feature = 0; feature < localFeatures.size(); feature++) {
                scaledLocal[ring][feature] = (float) artifacts.scaleFeature(
                    localFeatures.get(feature), rawLocal[ring][feature]
                );
            }
        }
        return new PredictionContext(
            input, communities, year, week, rawProperty, scaledProperty, rawTime,
            (float) artifacts.scaleFeature("time_trend", rawTime), rawMarket,
            scaledMarket, rawLocal, scaledLocal
        );
    }

    public List<PredictionContext> prepareAll(
            List<EmbeddingModel.BatchInput> inputs, boolean demonstrationMode) {
        List<PredictionContext> contexts = new ArrayList<>(inputs.size());
        for (EmbeddingModel.BatchInput input : inputs) {
            contexts.add(prepare(input, demonstrationMode));
        }
        return contexts;
    }

    private WaterProximityService.WaterFeatures water(EmbeddingModel.BatchInput input) {
        if (Double.isFinite(input.latitude()) && Double.isFinite(input.longitude())) {
            return waterProximity.lookup(input.latitude(), input.longitude());
        }
        return waterProximity.lookupH3(input.h3Index());
    }
}
