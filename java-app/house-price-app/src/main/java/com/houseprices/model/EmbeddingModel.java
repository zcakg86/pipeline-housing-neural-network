package com.houseprices.model;

import ai.onnxruntime.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.InputStream;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.time.temporal.IsoFields;
import java.util.List;
import java.util.Map;

/**
 * ONNX-backed embedding model for house price prediction.
 * Handles all feature engineering and inference.
 */
@ApplicationScoped
public class EmbeddingModel {

    private static final Logger LOG = Logger.getLogger(EmbeddingModel.class);

    // Default market indicators (can be updated via API)
    private static final double DEFAULT_MORTGAGE_RATE    = 6.5;
    private static final double DEFAULT_UNEMPLOYMENT_RATE = 4.0;

    @Inject ModelArtifacts artifacts;

    private OrtEnvironment env;
    private OrtSession    session;

    @PostConstruct
    void init() {
        try {
            env = OrtEnvironment.getEnvironment();
            byte[] modelBytes;
            try (InputStream is = getClass().getClassLoader()
                    .getResourceAsStream("model-artifacts/model.onnx")) {
                if (is == null) throw new RuntimeException("model.onnx not found in resources");
                modelBytes = is.readAllBytes();
            }
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
            LOG.info("ONNX model loaded successfully");
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    @PreDestroy
    void close() {
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (env != null) env.close(); }     catch (Exception ignored) {}
    }

    /**
     * Predict price for a single property.
     *
     * @param h3Index   H3 level-9 index string
     * @param saleDate  Date of sale / listing date
     * @param sqft      Living area sq ft
     * @param sqftLot   Lot size sq ft
     * @param beds      Bedrooms
     * @return predicted price in dollars
     */
    public double predict(String h3Index, LocalDate saleDate,
                          double sqft, double sqftLot, double beds) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds,
                       DEFAULT_MORTGAGE_RATE, DEFAULT_UNEMPLOYMENT_RATE);
    }

    public double predict(String h3Index, LocalDate saleDate,
                          double sqft, double sqftLot, double beds,
                          double mortgageRate, double unemploymentRate) {
        try {
            // ── Community neighbors (1 x 7) ───────────────────────────────
            int[] neighbors = artifacts.lookupH3Neighbors(h3Index);
            long[][] communityArr = new long[1][7];
            for (int i = 0; i < 7; i++) communityArr[0][i] = neighbors[i];

            // ── Year / week indices ───────────────────────────────────────
            int year = saleDate.getYear();
            int week = saleDate.get(IsoFields.WEEK_OF_WEEK_BASED_YEAR);
            long[] yearArr = {artifacts.lookupYear(year)};
            long[] weekArr = {artifacts.lookupWeek(week)};

            // ── Property features (scaled) ────────────────────────────────
            float[][] propArr = {{
                (float) artifacts.scaleFeature("sqft",     sqft),
                (float) artifacts.scaleFeature("sqft_lot", sqftLot),
                (float) artifacts.scaleFeature("beds",     beds)
            }};

            // ── Continuous time features (scaled) ─────────────────────────
            LocalDate refDate = artifacts.getReferenceDate();
            double timeTrend = ChronoUnit.DAYS.between(refDate, saleDate) / 365.25;
            double dayOfYear = saleDate.getDayOfYear();
            double sinDay    = Math.sin(2 * Math.PI * dayOfYear / 365.25);
            double cosDay    = Math.cos(2 * Math.PI * dayOfYear / 365.25);
            double sinMonth  = Math.sin(2 * Math.PI * saleDate.getMonthValue() / 12.0);
            double cosMonth  = Math.cos(2 * Math.PI * saleDate.getMonthValue() / 12.0);

            float[][] timeArr = {{
                (float) artifacts.scaleFeature("time_trend", timeTrend),
                (float) artifacts.scaleFeature("sin_day",    sinDay),
                (float) artifacts.scaleFeature("cos_day",    cosDay),
                (float) artifacts.scaleFeature("sin_month",  sinMonth),
                (float) artifacts.scaleFeature("cos_month",  cosMonth)
            }};

            // ── Market features (scaled) ──────────────────────────────────
            float[][] marketArr = {{
                (float) artifacts.scaleFeature("mortgage_rate",     mortgageRate),
                (float) artifacts.scaleFeature("unemployment_rate", unemploymentRate)
            }};

            // ── Run inference ─────────────────────────────────────────────
            try (OnnxTensor tCommunity = OnnxTensor.createTensor(env, communityArr);
                 OnnxTensor tYear      = OnnxTensor.createTensor(env, yearArr);
                 OnnxTensor tWeek      = OnnxTensor.createTensor(env, weekArr);
                 OnnxTensor tProp      = OnnxTensor.createTensor(env, propArr);
                 OnnxTensor tTime      = OnnxTensor.createTensor(env, timeArr);
                 OnnxTensor tMarket    = OnnxTensor.createTensor(env, marketArr)) {

                Map<String, OnnxTensor> inputs = Map.of(
                    "community_indices",  tCommunity,
                    "year",               tYear,
                    "week",               tWeek,
                    "property_features",  tProp,
                    "time_features",      tTime,
                    "market_features",    tMarket
                );

                try (OrtSession.Result result = session.run(inputs)) {
                    float[][] output = (float[][]) result.get(0).getValue();
                    double scaledLogPrice = output[0][0];
                    double logPrice = artifacts.inverseScaleLogPrice(scaledLogPrice);
                    return Math.exp(logPrice);
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException("ONNX inference failed", e);
        }
    }

    /**
     * Batch predict for a list of property records.
     */
    public double[] predictBatch(List<PropertyRecord> records) {
        double[] results = new double[records.size()];
        for (int i = 0; i < records.size(); i++) {
            PropertyRecord r = records.get(i);
            results[i] = predict(r.h3Index(), r.saleDate(), r.sqft(), r.sqftLot(), r.beds());
        }
        return results;
    }

    /** Simple value record for batch prediction */
    public record PropertyRecord(
        String h3Index, LocalDate saleDate,
        double sqft, double sqftLot, double beds
    ) {}
}
