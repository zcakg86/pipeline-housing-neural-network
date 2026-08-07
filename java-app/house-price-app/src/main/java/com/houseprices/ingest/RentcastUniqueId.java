package com.houseprices.ingest;

import java.math.BigInteger;
import java.util.Map;

/**
 * Builds the canonical RentCast sale identity used by every fetch path.
 *
 * <p>The identifier is the decimal integer formed from the numeric characters
 * in {@code lastSaleDate}'s first ten characters, {@code zipCode}, and
 * {@code assessorID}, in that order. {@link BigInteger} is necessary because
 * the combined value can exceed Java's {@code long} range.</p>
 */
public final class RentcastUniqueId {

    public static final String JSON_FIELD = "uniqueId";

    private RentcastUniqueId() {}

    /** Add the canonical numeric ID to one API property record, if derivable. */
    public static void addTo(Map<String, Object> property) {
        BigInteger id = from(property);
        if (id != null) {
            property.put(JSON_FIELD, id);
        }
    }

    /** Return the canonical numeric ID, or {@code null} when all inputs are empty. */
    public static BigInteger from(Map<String, Object> property) {
        String digits = digits(firstTen(property.get("lastSaleDate")))
            + digits(property.get("zipCode"))
            + digits(property.get("assessorID"));
        return digits.isEmpty() ? null : new BigInteger(digits);
    }

    /**
     * Convert a JSON numeric ID to the stable string key required by
     * {@link com.houseprices.service.PropertyStore}. The number must not be
     * converted through {@code double}, which would lose precision.
     */
    public static String storeKey(Object value) {
        if (value instanceof BigInteger id) return id.toString();
        if (value instanceof Number number) return new BigInteger(number.toString()).toString();
        String digits = digits(value);
        return digits.isEmpty() ? "" : new BigInteger(digits).toString();
    }

    private static String firstTen(Object value) {
        String text = value == null ? "" : String.valueOf(value);
        return text.substring(0, Math.min(text.length(), 10));
    }

    private static String digits(Object value) {
        String text = value == null ? "" : String.valueOf(value);
        StringBuilder result = new StringBuilder(text.length());
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            if (Character.isDigit(character)) result.append(character);
        }
        return result.toString();
    }
}
