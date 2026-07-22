package com.houseprices.model;

/** Prepared model inputs shared by neural and tree inference for one observation. */
public record PredictionContext(
    EmbeddingModel.BatchInput input,
    int[] communities,
    long yearIndex,
    long weekIndex,
    double[] rawProperty,
    float[] scaledProperty,
    double rawTimeTrend,
    float scaledTimeTrend,
    float[] rawMarket,
    float[] scaledMarket,
    float[][] rawLocalMarket,
    float[][] scaledLocalMarket
) {}
