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
 * The ONNX model exposes three outputs:
 *   0: log_price_scaled  [batch, 1]
 *   1: log_var_scaled    [batch, 1]   — uncertainty head (log variance in scaled log-price space)
 *   2: cls_attention     [batch, 6]   — CLS token attention over [community, year, week, property, time, market]
 */
@ApplicationScoped
public class EmbeddingModel {

    private static final Logger LOG = Logger.getLogger(EmbeddingModel.class);

    // Default market indicators (can be updated via API)
    private static final double DEFAULT_MORTGAGE_RATE     = 6.5;
    private static final double DEFAULT_UNEMPLOYMENT_RATE = 4.0;

    /** Token names matching the 6 CLS attention output positions */
    public static final String[] ATTENTION_TOKENS =
        {"community", "year", "week", "property", "time", "market"};

    /**
     * Full prediction result carrying price, uncertainty, and attention weights.
     *
     * @param predictedPrice     Price in dollars
     * @param predictionStdPrice Standard deviation in dollars
     *                           (delta method: std_price ≈ price × std_log_price)
     * @param predictionCvPct    95% CI width as % of predicted price
     *                           (3.92 × std_log_price × 100).
     *                           Price-normalised: 15% means the same regardless of
     *                           whether the property is $300k or $1.5m.
     *                           Directly interpretable as model confidence.
     * @param clsAttention       6-element attention weights over input tokens
     *                           [community, year, week, property, time, market]
     */
    public record PredictionResult(
        double  predictedPrice,
        double  predictionStdPrice,
        double  predictionCvPct,
        float[] clsAttention
    ) {
        /** Convenience: predictedPrice only, for callers that don't need extras */
        public static PredictionResult priceOnly(double price) {
            return new PredictionResult(price, 0.0, 0.0, new float[6]);
        }
    }

    @Inject ModelArtifacts artifacts;

    private OrtEnvironment env;
    private OrtSession     session;

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
            // Verify expected output count
            long nOutputs = session.getNumOutputs();
            LOG.infof("ONNX model loaded — %d output(s)", nOutputs);
            if (nOutputs < 3) {
                LOG.warn("Model has fewer than 3 outputs — uncertainty and attention will be zero. " +
                         "Re-run export_model_for_java.py to regenerate the ONNX file.");
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    @PreDestroy
    void close() {
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (env != null) env.close(); }         catch (Exception ignored) {}
    }

    // ── Public predict API ────────────────────────────────────────────────────

    /** Predict with default market indicators. */
    public PredictionResult predict(String h3Index, LocalDate saleDate,
                                    double sqft, double sqftLot, double beds) {
        return predict(h3Index, saleDate, sqft, sqftLot, beds,
                       DEFAULT_MORTGAGE_RATE, DEFAULT_UNEMPLOYMENT_RATE);
    }

    /** Predict with explicit market indicators. */
    public PredictionResult predict(String h3Index, LocalDate saleDate,
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
            LocalDate refDate  = artifacts.getReferenceDate();
            double    timeTrend = ChronoUnit.DAYS.between(refDate, saleDate) / 365.25;
            float[][] timeArr  = {{(float) artifacts.scaleFeature("time_trend", timeTrend)}};

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
                    // Output 0: log_price_scaled  [1, 1]
                    float[][] priceOut = (float[][]) result.get(0).getValue();
                    double scaledLogPrice = priceOut[0][0];
                    double logPrice       = artifacts.inverseScaleLogPrice(scaledLogPrice);
                    double predictedPrice = Math.exp(logPrice);

                    // Output 1: log_var_scaled  [1, 1]  — may be absent in old models
                    double predStdPrice = 0.0;
                    double predCvPct    = 0.0;
                    if (result.size() > 1) {
                        float[][] logVarOut = (float[][]) result.get(1).getValue();
                        double logVarScaled = logVarOut[0][0];
                        // std in unscaled log-price space
                        double logPriceScale = artifacts.getLogPriceScale();
                        double stdLogPrice   = Math.sqrt(Math.exp(logVarScaled)) * logPriceScale;
                        // Dollar std (delta method: std_price ≈ price × std_log_price)
                        predStdPrice = predictedPrice * stdLogPrice;
                        // 95% CI width as % of predicted price: 3.92 × std_log_price × 100
                        // This cancels the price factor — directly comparable across price levels
                        predCvPct = 3.92 * stdLogPrice * 100.0;
                    }

                    // Output 2: cls_attention  [1, 6]  — may be absent in old models
                    float[] clsAttn = new float[6];
                    if (result.size() > 2) {
                        float[][] attnOut = (float[][]) result.get(2).getValue();
                        System.arraycopy(attnOut[0], 0, clsAttn, 0, Math.min(6, attnOut[0].length));
                    }

                    return new PredictionResult(predictedPrice, predStdPrice, predCvPct, clsAttn);
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException("ONNX inference failed", e);
        }
    }

    // ── Batch predict ─────────────────────────────────────────────────────────

    /** Batch predict — returns a PredictionResult per record. */
    public PredictionResult[] predictBatch(List<BatchInput> records) {
        PredictionResult[] results = new PredictionResult[records.size()];
        for (int i = 0; i < records.size(); i++) {
            BatchInput r = records.get(i);
            results[i] = predict(r.h3Index(), r.saleDate(), r.sqft(), r.sqftLot(), r.beds());
        }
        return results;
    }

    /** Lightweight input record for batch prediction. */
    public record BatchInput(
        String h3Index, LocalDate saleDate,
        double sqft, double sqftLot, double beds
    ) {}
}
