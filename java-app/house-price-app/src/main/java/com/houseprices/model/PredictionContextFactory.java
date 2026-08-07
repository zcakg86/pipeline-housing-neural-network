package com.houseprices.model;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

/** Prepares date, spatial, water, market and local inputs exactly once. */
@ApplicationScoped
public class PredictionContextFactory {

    @Inject ModelArtifacts artifacts;
    @Inject WaterProximityService waterProximity;
    @Inject FeatureContract featureContract;
    @Inject RentcastLocalMarketService rentcastLocalMarket;

    public PredictionContext prepare(EmbeddingModel.BatchInput input, boolean demonstrationMode) {
        return prepare(input, demonstrationMode, false);
    }

    /** Prepare a displayed RentCast sale with the dedicated rolling market state. */
    public PredictionContext prepareRentcast(EmbeddingModel.BatchInput input) {
        return prepare(input, true, true);
    }

    private PredictionContext prepare(
            EmbeddingModel.BatchInput input, boolean demonstrationMode, boolean useRentcastMarket) {
        if (input.saleDate() == null) {
            throw new IllegalArgumentException("Prediction date is required");
        }
        int[] communities = artifacts.lookupH3Neighbors(input.h3Index());
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
                default -> throw new IllegalStateException("Unsupported property feature: " + name);
            };
            rawProperty[index] = value;
            scaledProperty[index] = (float) artifacts.scaleFeature(name, value);
        }
        double timeTrend = ChronoUnit.DAYS.between(
            artifacts.getReferenceDate(), input.saleDate()
        ) / 365.25;
        double annualPhase = 2.0 * Math.PI
            * (input.saleDate().getDayOfYear() - 1.0)
            / input.saleDate().lengthOfYear();
        double[] rawTime = {
            timeTrend, Math.sin(annualPhase), Math.cos(annualPhase)
        };
        List<String> timeFeatures = featureContract.timeFeatures();
        if (timeFeatures.size() != rawTime.length) {
            throw new IllegalStateException(
                "Expected time_trend, annual_sin and annual_cos in feature contract"
            );
        }
        float[] scaledTime = new float[rawTime.length];
        for (int index = 0; index < rawTime.length; index++) {
            scaledTime[index] = (float) artifacts.scaleFeature(
                timeFeatures.get(index), rawTime[index]
            );
        }
        float[] rawMarket = {
            (float) input.mortgageRate(), (float) input.unemploymentRate()
        };
        float[] scaledMarket = {
            (float) artifacts.scaleFeature("mortgage_rate", input.mortgageRate()),
            (float) artifacts.scaleFeature("unemployment_rate", input.unemploymentRate())
        };
        float[][] rawLocal = useRentcastMarket
            ? rentcastLocalMarket.rawFeatures(input.h3Index(), input.saleDate())
            : demonstrationMode
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
            input, communities, rawProperty, scaledProperty, rawTime, scaledTime, rawMarket,
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

    /** Batch variant used by ingestion so repeated snapshots are not model-scored one record at a time. */
    public List<PredictionContext> prepareRentcastAll(List<EmbeddingModel.BatchInput> inputs) {
        List<PredictionContext> contexts = new ArrayList<>(inputs.size());
        for (EmbeddingModel.BatchInput input : inputs) contexts.add(prepareRentcast(input));
        return contexts;
    }

    private WaterProximityService.WaterFeatures water(EmbeddingModel.BatchInput input) {
        if (Double.isFinite(input.latitude()) && Double.isFinite(input.longitude())) {
            return waterProximity.lookup(input.latitude(), input.longitude());
        }
        return waterProximity.lookupH3(input.h3Index());
    }
}
