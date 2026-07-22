package com.houseprices.ingest;

import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.LightGBMModel;
import com.houseprices.model.PredictionContext;
import com.houseprices.model.PredictionContextFactory;
import com.houseprices.service.PropertyStore;
import com.uber.h3core.H3Core;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.*;
import java.time.LocalDate;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

/**
 * Ingests historical sales CSV and canonical RentCast/Zillow JSON files.
 * Live JSON records are normalized first and inferred in model batches.
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
    @Inject LightGBMModel     lightgbm;
    @Inject com.houseprices.model.ModelArtifacts artifacts;
    @Inject PredictionContextFactory contextFactory;
    @Inject PropertyStore store;

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

                    // h3 index may be missing - fall back to computing from lat/lng
                    String h3Index = resolveH3(cols, idx, lat, lng);

                    double salePrice = parseDouble(cols, idx, "sale_price");
                    LocalDate saleDate = parseDate(cols, idx, "sale_date");
                    double sqft    = parseDouble(cols, idx, "sqft");
                    double sqftLot = parseDoubleOrDefault(cols, idx, "sqft_lot", 0);

                    // beds column name varies
                    double beds = parseDoubleOrDefault(cols, idx, "beds",
                                  parseDoubleOrDefault(cols, idx, "bed", 0));

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
                    // Unique ID
                    String id = buildUniqueRecordId(
                        "sales", col(cols, idx, "id"), lineNum, lat, lng
                    );
                    // Resolve community from H3 L8 index
                    String community = artifacts.lookupCommunity(h3Index);

                    double predicted;
                    double pctError;
                    double predictionStdPrice;
                    double predictionCvPct;
                    float[] clsAttention;
                    double lightgbmPredicted;
                    double lightgbmPctError;

                    if (idx.containsKey("predicted_price")) {
                        // Historical baseline predictions were generated in
                        // Python with row-specific causal local features. Do
                        // not run them through the current deployment snapshot.
                        predicted = parseDouble(cols, idx, "predicted_price");
                        pctError = parseDoubleOrDefault(
                            cols,
                            idx,
                            "pct_error",
                            salePrice > 0 ? 100.0 * (predicted - salePrice) / salePrice : 0
                        );
                        predictionStdPrice = parseDoubleOrDefault(
                            cols, idx, "prediction_std_price", 0
                        );
                        predictionCvPct = predicted > 0
                            ? 3.92 * predictionStdPrice / predicted * 100.0
                            : 0.0;
                        clsAttention = parseAttention(cols, idx);
                    } else {
                        EmbeddingModel.PredictionResult pred = model.predict(
                            h3Index, saleDate, sqft, sqftLot, beds, lat, lng
                        );
                        predicted = pred.predictedPrice();
                        pctError = salePrice > 0
                            ? 100.0 * (predicted - salePrice) / salePrice : 0;
                        predictionStdPrice = pred.predictionStdPrice();
                        predictionCvPct = pred.predictionCvPct();
                        clsAttention = pred.clsAttention();
                    }

                    if (idx.containsKey("lightgbm_predicted_price")) {
                        lightgbmPredicted = parseDouble(cols, idx, "lightgbm_predicted_price");
                        lightgbmPctError = parseDoubleOrDefault(
                            cols,
                            idx,
                            "lightgbm_pct_error",
                            salePrice > 0
                                ? 100.0 * (lightgbmPredicted - salePrice) / salePrice : 0
                        );
                    } else {
                        lightgbmPredicted = 0.0;
                        lightgbmPctError = 0.0;
                    }

                    records.add(new PropertyRecord(
                        id,
                        col(cols, idx, "address"),
                        "sales",
                        lat, lng, h3Index, community,
                        sqft, sqftLot, (int) beds, baths,
                        col(cols, idx, "home_type"),
                        saleDate, salePrice, 0.0, null,
                        predicted, pctError,
                        lightgbmPredicted, lightgbmPctError,
                        predictionStdPrice, predictionCvPct, clsAttention
                    ));
                } catch (Exception e) {
                    // Skip malformed rows silently
                }
            }
        }
        LOG.infof("Ingested %d sales records from %s", records.size(), file.getName());
        return records;
    }

    // ── Rentcast JSON (saved from API fetch) ─────────────────────────────────

    @SuppressWarnings("unchecked")
    public List<PropertyRecord> ingestRentcastJson(File file) throws IOException {
        LOG.infof("Ingesting Rentcast JSON: %s", file.getName());
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        Map<String, Object> root = mapper.readValue(file,
            mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
        List<Map<String, Object>> properties = (List<Map<String, Object>>) root.get("properties");
        if (properties == null) return List.of();

        List<PendingProperty> pending = new ArrayList<>();
        for (Map<String, Object> p : properties) {
            try {
                double lat = toDouble(p.get("latitude"));
                double lng = toDouble(p.get("longitude"));
                String h3Index = h3.h3ToString(h3.latLngToCell(lat, lng, 8));

                double salePrice = toDoubleOrDefault(p.get("lastSalePrice"), 0);
                LocalDate saleDate = parseDateOrDefault(p.get("lastSaleDate"));
                double sqft    = toDoubleOrDefault(p.get("squareFootage"), 0);
                double sqftLot = toDoubleOrDefault(p.get("lotSize"), 0);
                int beds       = (int) toDoubleOrDefault(p.get("bedrooms"), 0);
                double baths   = toDoubleOrDefault(p.get("bathrooms"), 0);
                String address = stringValue(p.get("formattedAddress"));
                String id = stringValue(p.get("id"));
                if (id.isBlank()) id = address + "|" + saleDate;
                pending.add(new PendingProperty(
                    id, address, "rentcast", lat, lng, h3Index,
                    artifacts.lookupCommunity(h3Index), sqft, sqftLot, beds, baths,
                    stringValue(p.get("propertyType")), saleDate, salePrice, 0.0, null
                ));
            } catch (Exception e) {
                LOG.debugf("Skipping malformed RentCast entry: %s", e.getMessage());
            }
        }
        List<PropertyRecord> records = predictChanged(pending);
        LOG.infof("Ingested %d changed RentCast records from %s (%d parsed)",
            records.size(), file.getName(), pending.size());
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

        LocalDate observationDate = jsonObservationDate(root, file);
        List<PendingProperty> pending = new ArrayList<>();
        for (Map<String, Object> p : properties) {
            try {
                pending.add(zillowMapToPending(p, observationDate));
            } catch (Exception e) {
                LOG.debugf("Skipping malformed Zillow entry: %s", e.getMessage());
            }
        }
        List<PropertyRecord> records = predictChanged(pending);
        LOG.infof("Ingested %d changed Zillow listings from %s (%d parsed)",
            records.size(), file.getName(), pending.size());
        return records;
    }

    private record PendingProperty(
        String id, String address, String source,
        double lat, double lng, String h3Index, String community,
        double sqft, double sqftLot, int beds, double baths, String homeType,
        LocalDate saleDate, double salePrice, double zestimate, String listingUrl
    ) {}

    private List<PropertyRecord> predictChanged(List<PendingProperty> parsed) {
        Map<String, PendingProperty> unique = new LinkedHashMap<>();
        for (PendingProperty value : parsed) {
            unique.put(value.source() + "\u0000" + value.id(), value);
        }
        LocalDate latestSnapshotSale = artifacts.getLocalMarketLatestSaleDate();
        List<PendingProperty> changed = unique.values().stream()
            .filter(value -> latestSnapshotSale == null
                || value.saleDate().isAfter(latestSnapshotSale))
            .filter(value -> store.findBySourceAndId(value.source(), value.id())
                .map(existing -> !sameModelInputs(existing, value))
                .orElse(true))
            .toList();
        if (changed.isEmpty()) return List.of();

        List<EmbeddingModel.BatchInput> inputs = changed.stream()
            .map(value -> new EmbeddingModel.BatchInput(
                value.h3Index(), value.saleDate(), value.sqft(), value.sqftLot(),
                value.beds(), value.lat(), value.lng()
            ))
            .toList();
        List<PredictionContext> contexts = contextFactory.prepareAll(inputs, false);
        EmbeddingModel.PredictionResult[] neural = model.predictPrepared(contexts);
        double[] tree = lightgbm.predictPrepared(contexts);
        List<PropertyRecord> records = new ArrayList<>(changed.size());
        for (int index = 0; index < changed.size(); index++) {
            PendingProperty value = changed.get(index);
            EmbeddingModel.PredictionResult prediction = neural[index];
            double neuralError = value.salePrice() > 0
                ? 100.0 * (prediction.predictedPrice() - value.salePrice()) / value.salePrice()
                : 0.0;
            double treeError = value.salePrice() > 0
                ? 100.0 * (tree[index] - value.salePrice()) / value.salePrice()
                : 0.0;
            records.add(new PropertyRecord(
                value.id(), value.address(), value.source(), value.lat(), value.lng(),
                value.h3Index(), value.community(), value.sqft(), value.sqftLot(),
                value.beds(), value.baths(), value.homeType(), value.saleDate(),
                value.salePrice(), value.zestimate(), value.listingUrl(), prediction.predictedPrice(),
                neuralError, tree[index], treeError, prediction.predictionStdPrice(),
                prediction.predictionCvPct(), prediction.clsAttention()
            ));
        }
        return records;
    }

    private boolean sameModelInputs(PropertyRecord existing, PendingProperty value) {
        return Objects.equals(existing.address(), value.address())
            && Double.compare(existing.lat(), value.lat()) == 0
            && Double.compare(existing.lng(), value.lng()) == 0
            && Objects.equals(existing.h3Index(), value.h3Index())
            && Double.compare(existing.sqft(), value.sqft()) == 0
            && Double.compare(existing.sqftLot(), value.sqftLot()) == 0
            && existing.beds() == value.beds()
            && Double.compare(existing.baths(), value.baths()) == 0
            && Objects.equals(existing.homeType(), value.homeType())
            && Objects.equals(existing.saleDate(), value.saleDate())
            && Double.compare(existing.salePrice(), value.salePrice()) == 0
            && Double.compare(existing.zestimate(), value.zestimate()) == 0
            && Objects.equals(existing.listingUrl(), value.listingUrl());
    }

    private LocalDate jsonObservationDate(Map<String, Object> root, File file) {
        Object metadataValue = root.get("requestMetadata");
        if (metadataValue instanceof Map<?, ?> metadata) {
            Object fetchedAt = metadata.get("fetchedAt");
            if (fetchedAt != null) {
                try {
                    return Instant.parse(String.valueOf(fetchedAt))
                        .atZone(ZoneId.systemDefault()).toLocalDate();
                } catch (Exception ignored) {}
            }
        }
        return Instant.ofEpochMilli(file.lastModified())
            .atZone(ZoneId.systemDefault()).toLocalDate();
    }

    private PendingProperty zillowMapToPending(
            Map<String, Object> p, LocalDate observationDate) {
        double lat = toDouble(p.get("latitude"));
        double lng = toDouble(p.get("longitude"));
        String h3Index = h3.h3ToString(h3.latLngToCell(lat, lng, 8));
        double sqft = toDoubleOrDefault(p.get("area"),
            toDoubleOrDefault(p.get("livingArea"), 1500));
        double sqftLot = toDoubleOrDefault(p.get("lotAreaValue"), 5000);
        int beds = (int) toDoubleOrDefault(p.get("beds"),
            toDoubleOrDefault(p.get("bedrooms"), 3));
        double baths = toDoubleOrDefault(p.get("baths"),
            toDoubleOrDefault(p.get("bathrooms"), 2));
        double price = toDoubleOrDefault(p.get("price"), 0);
        double zestimate = toDoubleOrDefault(p.get("zestimate"), 0);
        String url = stringValue(p.getOrDefault("url", p.get("detailUrl")));
        String address;
        Object addressValue = p.get("address");
        if (addressValue instanceof Map<?, ?> addressMap) {
            address = stringValue(addressMap.get("street")) + ", "
                + stringValue(addressMap.get("city")) + ", "
                + stringValue(addressMap.get("state"));
        } else {
            address = stringValue(addressValue);
        }
        String id = stringValue(p.getOrDefault("id", p.get("zpid")));
        if (id.isBlank()) id = address;
        return new PendingProperty(
            id, address, "zillow", lat, lng, h3Index, artifacts.lookupCommunity(h3Index),
            sqft, sqftLot, beds, baths, stringValue(p.get("homeType")),
            observationDate, price, zestimate, url
        );
    }

    private String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private String resolveH3(String[] cols, Map<String, Integer> idx, double lat, double lng) {
        // Try known h3 column names in order of preference
        for (String col : new String[]{"h3_09", "h3_10", "h3_08", "h3_07"}) {
            String existing = col(cols, idx, col);
            if (!existing.isBlank()) {
                // If it's not level 8, convert to level 8 parent/child
                try {
                    long cellLong = h3.stringToH3(existing);
                    int res = h3.getResolution(cellLong);
                    if (res == 8) return existing;
                    // For any other resolution just compute from lat/lng
                } catch (Exception ignored) {}
            }
        }
        return h3.h3ToString(h3.latLngToCell(lat, lng, 8));
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

    private float[] parseAttention(String[] cols, Map<String, Integer> idx) {
        String[] names = {
            "cls_attn_community", "cls_attn_year", "cls_attn_week",
            "cls_attn_property", "cls_attn_time", "cls_attn_market"
        };
        float[] attention = new float[names.length];
        for (int i = 0; i < names.length; i++) {
            attention[i] = (float) parseDoubleOrDefault(cols, idx, names[i], 0.0);
        }
        return attention;
    }

    private LocalDate parseDate(String[] cols, Map<String, Integer> idx, String name) {
        return parseLocalDate(col(cols, idx, name));
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

    static String buildUniqueRecordId(
        String source, String preferredId, int rowNumber, double lat, double lng
    ) {
        String stablePart = preferredId == null || preferredId.isBlank()
            ? String.format(java.util.Locale.ROOT, "%.5f_%.5f", lat, lng)
            : preferredId.trim();
        return source + "_" + stablePart + "_" + rowNumber;
    }

    private LocalDate parseDateOrDefault(Object o) {
        try { return parseLocalDate(String.valueOf(o)); }
        catch (Exception e) { return LocalDate.now(); }
    }

}
