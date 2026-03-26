package com.houseprices.ingest;

import java.time.LocalDate;

/**
 * Unified property record used across ingestion, prediction and serving.
 */
public record PropertyRecord(
    // Identity
    String  id,
    String  address,
    String  source,          // "zillow" | "rentcast" | "sales"

    // Location
    double  lat,
    double  lng,
    String  h3Index,         // H3 level-9 index

    // Property attributes
    double  sqft,
    double  sqftLot,
    int     beds,
    double  baths,
    String  homeType,

    // Sale / listing
    LocalDate saleDate,
    double    salePrice,     // 0 if unknown (Zillow listing)
    String    listingUrl,    // Zillow URL if applicable

    // Predictions (populated after inference)
    double  predictedPrice,
    double  pctError         // 0 if salePrice unknown
) {
    /** Convenience: does this record have an actual sale price to compare against? */
    public boolean hasSalePrice() { return salePrice > 0; }
}
