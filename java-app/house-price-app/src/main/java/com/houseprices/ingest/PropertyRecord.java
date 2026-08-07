package com.houseprices.ingest;

import java.time.LocalDate;

/**
 * Unified property record used across ingestion, prediction and serving.
 *
 * New inference fields (populated after ONNX inference):
 *   predictionStdPrice  — prediction standard deviation in $ (from uncertainty head)
 *   clsAttention        — four CLS attention weights over
 *                         [community, property, time, market]
 */
public record PropertyRecord(
    // Identity
    String  id,
    String  address,
    String  source,          // "zillow" | "rentcast" | "sales"

    // Location
    double  lat,
    double  lng,
    String  h3Index,         // H3 index
    String  community,       // Community ID (from H3 → community map)

    // Property attributes
    double  sqft,
    double  sqftLot,
    int     beds,
    double  baths,
    String  homeType,

    // Sale / listing
    LocalDate saleDate,
    double    salePrice,         // 0 if unknown (Zillow listing)
    double    zestimate,         // Zillow estimate, 0 if unavailable/not applicable
    String    listingUrl,        // Zillow URL if applicable

    // Predictions (populated after inference)
    double    predictedPrice,
    double    pctError,          // 0 if salePrice unknown
    double    lightgbmPredictedPrice,
    double    lightgbmPctError,  // 0 if salePrice unknown
    double    gnnPredictedPrice,
    double    gnnPctError,       // 0 if salePrice unknown

    // Uncertainty and attention (populated after inference)
    double    predictionStdPrice, // std dev of prediction in $ (0 if unavailable)
    double    predictionCvPct,    // 95% CI width as % of predicted price (0 if unavailable)
    float[]   clsAttention        // [community, property, time, market], null if unavailable
) {
    /** Convenience: does this record have an actual sale price to compare against? */
    public boolean hasSalePrice() { return salePrice > 0; }
}
