package com.houseprices.service;

import com.houseprices.ingest.PropertyRecord;

import java.time.LocalDate;
import java.time.format.DateTimeParseException;

/**
 * Immutable, validated filters shared by property-serving endpoints and H3
 * aggregation. Parsing happens once per request so per-record predicates only
 * perform comparisons and never parse dates or reinterpret partial bounds.
 */
public record PropertyRequestFilter(
    String homeType,
    LocalDate dateFrom,
    LocalDate dateTo,
    double minError,
    double maxError,
    GeoBounds bounds
) {
    /** Geographic viewport; {@code unbounded=true} represents no map constraint. */
    public record GeoBounds(
        double west,
        double south,
        double east,
        double north,
        boolean unbounded
    ) {
        boolean includes(PropertyRecord record) {
            return unbounded
                || (record.lng() >= west && record.lng() <= east
                    && record.lat() >= south && record.lat() <= north);
        }
    }

    /**
     * Parse query parameters and reject ambiguous date ranges, inverted error
     * ranges, and partial or invalid geographic viewports.
     */
    public static PropertyRequestFilter parse(
            String homeType,
            String rawDateFrom,
            String rawDateTo,
            double minError,
            double maxError,
            Double west,
            Double south,
            Double east,
            Double north) {
        LocalDate dateFrom = parseDate(rawDateFrom, "dateFrom");
        LocalDate dateTo = parseDate(rawDateTo, "dateTo");
        if (dateFrom != null && dateTo != null && dateFrom.isAfter(dateTo)) {
            throw new IllegalArgumentException("dateFrom must be on or before dateTo");
        }
        if (minError > maxError) {
            throw new IllegalArgumentException("minError must be at most maxError");
        }
        return new PropertyRequestFilter(
            homeType == null || homeType.isBlank() ? "all" : homeType,
            dateFrom,
            dateTo,
            minError,
            maxError,
            parseBounds(west, south, east, north)
        );
    }

    /** Return whether a record satisfies every common request constraint. */
    public boolean includes(PropertyRecord record, double selectedError) {
        if (selectedError < minError || selectedError > maxError) return false;
        if (!"all".equalsIgnoreCase(homeType)) {
            String type = record.homeType();
            if (type == null || type.isBlank() || !homeType.equalsIgnoreCase(type)) return false;
        }
        if (!includesDate(record)) return false;
        return bounds.includes(record);
    }

    private boolean includesDate(PropertyRecord record) {
        if (dateFrom == null && dateTo == null) return true;
        LocalDate date = record.saleDate();
        if (date == null) return false;
        return (dateFrom == null || !date.isBefore(dateFrom))
            && (dateTo == null || !date.isAfter(dateTo));
    }

    private static LocalDate parseDate(String raw, String name) {
        if (raw == null || raw.isBlank()) return null;
        try {
            return LocalDate.parse(raw);
        } catch (DateTimeParseException exception) {
            throw new IllegalArgumentException(name + " must use YYYY-MM-DD", exception);
        }
    }

    private static GeoBounds parseBounds(
            Double west, Double south, Double east, Double north) {
        boolean none = west == null && south == null && east == null && north == null;
        if (none) return new GeoBounds(0, 0, 0, 0, true);
        if (west == null || south == null || east == null || north == null
                || west > east || south > north
                || west < -180 || east > 180 || south < -90 || north > 90) {
            throw new IllegalArgumentException(
                "west, south, east and north must form a valid bounding box"
            );
        }
        return new GeoBounds(west, south, east, north, false);
    }
}
