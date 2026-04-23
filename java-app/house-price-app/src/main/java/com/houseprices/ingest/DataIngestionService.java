package com.houseprices.ingest;

import com.houseprices.model.EmbeddingModel;
import com.uber.h3core.H3Core;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

/**
 * Ingests property data from CSV files (Rentcast drops, sales history, Zillow)
 * and directly from the Rentcast / Zillow APIs.
 * One method per source type; all return List<PropertyRecord>.
 */
@ApplicationScoped
public class DataIngestionService {

    private static final Logger LOG = Logger.getLogger(DataIngestionService.class);
    private static final DateTimeFormatter[] DATE_FORMATS = {
        DateTimeFormatter.ofPattern("yyyy-MM-dd"),
        DateTimeFormatter.ofPattern("MM/dd/yyyy"),
        DateTimeFormatter.ofPattern("M/d/yyyy")
    };

    @Inject EmbeddingModel    model;
    @Inject RentcastApiClient rentcastApi;
    @Inject ZillowApiClient   zillowApi;
    @Inject com.houseprices.model.ModelArtifacts artifacts;

    private final H3Core h3;

    public DataIngestionService() {
        try { h3 = H3Core.newInstance(); }
        catch (IOException e) { throw new RuntimeException("H3Core init failed", e); }
    }

    // ── Sales history CSV (data/sales_2020_25.csv format) ────────────────────

    public List<PropertyRecord> ingestSalesCsv(File file) throws IOException {
        LOG.infof("Ingesting sales CSV: %s", file.getName());
        List<PropertyRecord> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String headerLine = br.readLine();
            if (headerLine == null) return records;
            Map<String, Integer> idx = headerIndex(headerLine);

            String line;
            int lineNum = 0;
            while ((line = br.readLine()) != null) {
                lineNum++;
                String[] cols = parseCsvLine(line);
                try {
                    double lat = parseDouble(cols, idx, "lat");
                    double lng = parseDouble(cols, idx, "lng");

                    // h3_09 may be missing - fall back to computing from lat/lng
                    String h3Index = resolveH3(cols, idx, lat, lng);

                    double salePrice = parseDouble(cols, idx, "sale_price");
                    LocalDate saleDate = parseDate(cols, idx, "sale_date");
                    double sqft    = parseDouble(cols, idx, "sqft");
                    double sqftLot = parseDoubleOrDefault(cols, idx, "sqft_lot", 5000);

                    // beds column name varies
                    double beds = parseDoubleOrDefault(cols, idx, "beds",
                                  parseDoubleOrDefault(cols, idx, "bed", 3));

                    // baths: prefer combined 'baths', else sum bath_full + bath_3qtr + bath_half
                    double baths;
                    if (idx.containsKey("baths")) {
                        baths = parseDoubleOrDefault(cols, idx, "baths", 2);
                    } else {
                        double full  = parseDoubleOrDefault(cols, idx, "bath_full",  0);
                        double three = parseDoubleOrDefault(cols, idx, "bath_3qtr",  0);
                        double half  = parseDoubleOrDefault(cols, idx, "bath_half",  0);
                        baths = full + three * 0.75 + half * 0.5;
                        if (baths == 0) baths = 2;
                    }

                    // sale_nbr used as id
                    String id = col(cols, idx, "sale_nbr");
                    if (id.isBlank()) id = lat + "_" + lng + "_" + lineNum;

                    // Resolve community from H3 L9 index
                    String community = artifacts.lookupCommunity(h3Index);

                    double predicted = model.predict(h3Index, saleDate, sqft, sqftLot, beds);
                    double pctError  = salePrice > 0
                        ? 100.0 * (predicted - salePrice) / salePrice : 0;

                    records.add(new PropertyRecord(
                        id,
                        col(cols, idx, "address"),
                        "sales",
                        lat, lng, h3Index, community,
                        sqft, sqftLot, (int) beds, baths,
                        col(cols, idx, "home_type"),
                        saleDate, salePrice, null,
                        predicted, pctError
                    ));
                } catch (Exception e) {
                    // Skip malformed rows silently
                }
            }
        }
        LOG.infof("Ingested %d sales records from %s", records.size(), file.getName());
        return records;
    }

    // ── Rentcast CSV (data/rentcast_recent_house_sales.csv format) ───────────

    public List<PropertyRecord> ingestRentcastCsv(File file) throws IOException {
        LOG.infof("Ingesting Rentcast CSV: %s", file.getName());
        List<PropertyRecord> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String headerLine = br.readLine();
            if (headerLine == null) return records;
            Map<String, Integer> idx = headerIndex(headerLine);

            String line;
            while ((line = br.readLine()) != null) {
                String[] cols = parseCsvLine(line);
                try {
                    double lat = parseDouble(cols, idx, "latitude");
                    double lng = parseDouble(cols, idx, "longitude");
                    String h3Index = h3.h3ToString(h3.latLngToCell(lat, lng, 9));
                    double salePrice = parseDoubleOrDefault(cols, idx, "lastSalePrice", 0);
                    LocalDate saleDate = parseDateOrToday(cols, idx, "lastSaleDate");
                    double sqft    = parseDoubleOrDefault(cols, idx, "squareFootage", 1500);
                    double sqftLot = parseDoubleOrDefault(cols, idx, "lotSize", 5000);
                    double beds    = parseDoubleOrDefault(cols, idx, "bedrooms", 3);

                    double predicted = model.predict(h3Index, saleDate, sqft, sqftLot, beds);
                    double pctError  = salePrice > 0
                        ? 100.0 * (predicted - salePrice) / salePrice : 0;

                    records.add(new PropertyRecord(
                        col(cols, idx, "id"),
                        col(cols, idx, "formattedAddress"),
                        "rentcast",
                        lat, lng, h3Index, artifacts.lookupCommunity(h3Index),
                        sqft, sqftLot, (int) beds,
                        parseDoubleOrDefault(cols, idx, "bathrooms", 2),
                        col(cols, idx, "propertyType"),
                        saleDate, salePrice, null,
                        predicted, pctError
                    ));
                } catch (Exception e) {
                    // Skip malformed rows
                }
            }
        }
        LOG.infof("Ingested %d Rentcast records from %s", records.size(), file.getName());
        return records;
    }

    // ── Zillow listings JSON (data/zillow_seattle_listings.json format) ───────

    @SuppressWarnings("unchecked")
    public List<PropertyRecord> ingestZillowJson(File file) throws IOException {
        LOG.infof("Ingesting Zillow JSON: %s", file.getName());
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        Map<String, Object> root = mapper.readValue(file,
            mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
        List<Map<String, Object>> properties = (List<Map<String, Object>>) root.get("properties");
        if (properties == null) return List.of();

        List<PropertyRecord> records = new ArrayList<>();
        for (Map<String, Object> p : properties) {
            try {
                records.add(zillowMapToRecord(p));
            } catch (Exception e) {
                // Skip malformed entries
            }
        }
        LOG.infof("Ingested %d Zillow listings from %s", records.size(), file.getName());
        return records;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private String resolveH3(String[] cols, Map<String, Integer> idx, double lat, double lng) {
        // Try known h3 column names in order of preference
        for (String col : new String[]{"h3_09", "h3_10", "h3_08", "h3_07"}) {
            String existing = col(cols, idx, col);
            if (!existing.isBlank()) {
                // If it's not level 9, convert to level 9 parent/child
                try {
                    long cellLong = h3.stringToH3(existing);
                    int res = h3.getResolution(cellLong);
                    if (res == 9) return existing;
                    // For any other resolution just compute from lat/lng
                } catch (Exception ignored) {}
            }
        }
        return h3.h3ToString(h3.latLngToCell(lat, lng, 9));
    }

    private Map<String, Integer> headerIndex(String headerLine) {
        String[] headers = parseCsvLine(headerLine);
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < headers.length; i++) map.put(headers[i].trim(), i);
        return map;
    }

    private String col(String[] cols, Map<String, Integer> idx, String name) {
        Integer i = idx.get(name);
        return (i != null && i < cols.length) ? cols[i].trim() : "";
    }

    private double parseDouble(String[] cols, Map<String, Integer> idx, String name) {
        return Double.parseDouble(col(cols, idx, name));
    }

    private double parseDoubleOrDefault(String[] cols, Map<String, Integer> idx, String name, double def) {
        try { return Double.parseDouble(col(cols, idx, name)); } catch (Exception e) { return def; }
    }

    private LocalDate parseDate(String[] cols, Map<String, Integer> idx, String name) {
        return parseLocalDate(col(cols, idx, name));
    }

    private LocalDate parseDateOrToday(String[] cols, Map<String, Integer> idx, String name) {
        try { return parseLocalDate(col(cols, idx, name)); } catch (Exception e) { return LocalDate.now(); }
    }

    private LocalDate parseLocalDate(String s) {
        if (s == null || s.isBlank()) throw new DateTimeParseException("empty", s, 0);
        for (DateTimeFormatter fmt : DATE_FORMATS) {
            try { return LocalDate.parse(s.substring(0, Math.min(s.length(), 10)), fmt); }
            catch (Exception ignored) {}
        }
        throw new DateTimeParseException("Cannot parse date: " + s, s, 0);
    }

    private double toDouble(Object o) {
        if (o == null) throw new NumberFormatException("null");
        return ((Number) o).doubleValue();
    }

    private double toDoubleOrDefault(Object o, double def) {
        try { return toDouble(o); } catch (Exception e) { return def; }
    }

    /** Minimal CSV line parser (handles quoted fields) */
    private String[] parseCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;
        for (char c : line.toCharArray()) {
            if (c == '"') { inQuotes = !inQuotes; }
            else if (c == ',' && !inQuotes) { tokens.add(sb.toString()); sb.setLength(0); }
            else { sb.append(c); }
        }
        tokens.add(sb.toString());
        return tokens.toArray(new String[0]);
    }

    // ── API fetch methods ─────────────────────────────────────────────────────

    /**
     * Fetch recent sales from the RentCast API and return as PropertyRecords.
     * Requires rentcast.api.key to be set in application.properties.
     */
    public List<PropertyRecord> fetchFromRentcastApi() throws Exception {
        return fetchFromRentcastApi(0);
    }

    public List<PropertyRecord> fetchFromRentcastApi(int limitOverride) throws Exception {
        List<Map<String, Object>> raw = rentcastApi.fetchRecentSales(limitOverride);
        List<PropertyRecord> records = new ArrayList<>();
        for (Map<String, Object> p : raw) {
            try {
                double lat = toDouble(p.get("latitude"));
                double lng = toDouble(p.get("longitude"));
                String h3Index = h3.h3ToString(h3.latLngToCell(lat, lng, 9));

                double salePrice = toDoubleOrDefault(p.get("lastSalePrice"), 0);
                LocalDate saleDate = parseDateOrDefault(p.get("lastSaleDate"));
                double sqft    = toDoubleOrDefault(p.get("squareFootage"), 1500);
                double sqftLot = toDoubleOrDefault(p.get("lotSize"), 5000);
                double beds    = toDoubleOrDefault(p.get("bedrooms"), 3);
                double baths   = toDoubleOrDefault(p.get("bathrooms"), 2);

                double predicted = model.predict(h3Index, saleDate, sqft, sqftLot, beds);
                double pctError  = salePrice > 0 ? 100.0 * (predicted - salePrice) / salePrice : 0;

                records.add(new PropertyRecord(
                    String.valueOf(p.getOrDefault("id", "")),
                    String.valueOf(p.getOrDefault("formattedAddress", "")),
                    "rentcast",
                    lat, lng, h3Index, artifacts.lookupCommunity(h3Index),
                    sqft, sqftLot, (int) beds, baths,
                    String.valueOf(p.getOrDefault("propertyType", "")),
                    saleDate, salePrice, null,
                    predicted, pctError
                ));
            } catch (Exception e) {
                // Skip malformed entries
            }
        }
        LOG.infof("Transformed %d RentCast API records", records.size());
        return records;
    }

    /**
     * Fetch current listings from the Zillow API and return as PropertyRecords.
     * Requires zillow.api.key to be set in application.properties.
     */
    public List<PropertyRecord> fetchFromZillowApi() throws Exception {
        List<Map<String, Object>> raw = zillowApi.fetchListings();
        List<PropertyRecord> records = new ArrayList<>();
        for (Map<String, Object> p : raw) {
            try {
                records.add(zillowMapToRecord(p));
            } catch (Exception e) {
                // Skip malformed entries
            }
        }
        LOG.infof("Transformed %d Zillow API records", records.size());
        return records;
    }

    /** Shared conversion from a raw Zillow property map to a PropertyRecord */
    @SuppressWarnings("unchecked")
    private PropertyRecord zillowMapToRecord(Map<String, Object> p) {
        double lat = toDouble(p.get("latitude"));
        double lng = toDouble(p.get("longitude"));
        String h3Index = h3.h3ToString(h3.latLngToCell(lat, lng, 9));

        // sqft: API uses "area", file may also use "livingArea"
        double sqft    = toDoubleOrDefault(p.get("area"),
                         toDoubleOrDefault(p.get("livingArea"), 1500));
        double sqftLot = toDoubleOrDefault(p.get("lotAreaValue"), 5000);
        double beds    = toDoubleOrDefault(p.get("beds"),
                         toDoubleOrDefault(p.get("bedrooms"), 3));
        double baths   = toDoubleOrDefault(p.get("baths"),
                         toDoubleOrDefault(p.get("bathrooms"), 2));
        double price   = toDoubleOrDefault(p.get("price"), 0);

        // url: API uses "url", file may use "detailUrl"
        String url = String.valueOf(p.getOrDefault("url",
                     p.getOrDefault("detailUrl", "")));

        // address: may be a nested map {street, city, state} or a plain string
        String address;
        Object addrObj = p.get("address");
        if (addrObj instanceof Map) {
            Map<String, Object> addrMap = (Map<String, Object>) addrObj;
            address = addrMap.getOrDefault("street", "") + ", "
                    + addrMap.getOrDefault("city", "") + ", "
                    + addrMap.getOrDefault("state", "");
        } else {
            address = String.valueOf(addrObj != null ? addrObj : "");
        }

        String id = String.valueOf(p.getOrDefault("id",
                    p.getOrDefault("zpid", address)));

        double predicted = model.predict(h3Index, LocalDate.now(), sqft, sqftLot, beds);
        double pctError  = price > 0 ? 100.0 * (predicted - price) / price : 0;

        return new PropertyRecord(
            id, address, "zillow",
            lat, lng, h3Index, artifacts.lookupCommunity(h3Index),
            sqft, sqftLot, (int) beds, baths,
            String.valueOf(p.getOrDefault("homeType", "")),
            LocalDate.now(), price, url,
            predicted, pctError
        );
    }

    private LocalDate parseDateOrDefault(Object o) {
        try { return parseLocalDate(String.valueOf(o)); }
        catch (Exception e) { return LocalDate.now(); }
    }

    /** Expose API clients for callers that need the raw response (e.g. to save to disk) */
    public ZillowApiClient getZillowApi()   { return zillowApi; }
    public RentcastApiClient getRentcastApi() { return rentcastApi; }
}
