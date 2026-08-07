package com.houseprices.model;

/** Prepared model inputs shared by neural and tree inference for one observation. */
public record PredictionContext(
    EmbeddingModel.BatchInput input,
    int[] communities,
    double[] rawProperty,
    float[] scaledProperty,
    double[] rawTime,
    float[] scaledTime,
    float[] rawMarket,
    float[] scaledMarket,
    float[][] rawLocalMarket,
    float[][] scaledLocalMarket
) {}
