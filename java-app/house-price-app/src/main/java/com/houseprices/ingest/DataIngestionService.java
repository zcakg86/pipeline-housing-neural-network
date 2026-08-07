package com.houseprices.ingest;

import com.houseprices.model.EmbeddingModel;
import com.houseprices.model.LightGBMModel;
import com.houseprices.model.GnnModel;
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
    @Inject GnnModel          gnn;
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
        long started = System.nanoTime();
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
                    String id = buildUniqueRecordId(saleDate, lineNum);
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
                        lightgbmPredicted, lightgbmPctError, 0.0, 0.0,
                        predictionStdPrice, predictionCvPct, clsAttention
                    ));
                } catch (Exception e) {
                    // Skip malformed rows silently
                }
            }
        }
        long gnnStarted = System.nanoTime();
        records = withGnnPredictions(records);
        long gnnElapsed = System.nanoTime() - gnnStarted;
        LOG.infof("Historical sales ingestion: %,d records — CSV parse %.1f ms, GNN %.1f ms, total %.1f ms",
            records.size(), (gnnStarted - started) / 1_000_000.0,
            gnnElapsed / 1_000_000.0, (System.nanoTime() - started) / 1_000_000.0);
        LOG.infof("Ingested %d sales records from %s", records.size(), file.getName());
        return records;
    }

    // ── Rentcast JSON (saved from API fetch) ─────────────────────────────────

    @SuppressWarnings("unchecked")
    public List<PropertyRecord> ingestRentcastJson(File file) throws IOException {
        return ingestRentcastJsonFiles(List.of(file));
    }

    /** Parse all saved RentCast responses, then score the de-duplicated source in one batch. */
    @SuppressWarnings("unchecked")
    public List<PropertyRecord> ingestRentcastJsonFiles(List<File> files) throws IOException {
        LOG.infof("Ingesting %d RentCast JSON file(s) as one batch", files.size());
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        List<PendingProperty> pending = new ArrayList<>();
        for (File file : files) {
            List<Map<String, Object>> properties;
            try {
                Map<String, Object> root = mapper.readValue(file,
                    mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
                properties = (List<Map<String, Object>>) root.get("properties");
            } catch (IOException exception) {
                LOG.warnf("Skipping unreadable RentCast JSON %s: %s", file.getName(), exception.getMessage());
                continue;
            }
            if (properties == null) continue;
            for (Map<String, Object> p : properties) try {
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
                // RentCast JSON is written with a decimal integer uniqueId.
                // Keep its exact decimal representation as the internal map
                // key: a Java int/long cannot hold all assessor-derived IDs.
                String id = RentcastUniqueId.storeKey(p.get(RentcastUniqueId.JSON_FIELD));
                if (id.isBlank()) {
                    throw new IllegalArgumentException("RentCast record is missing numeric uniqueId");
                }
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
        LOG.infof("Ingested %d changed RentCast records (%d parsed from %d file(s))",
            records.size(), pending.size(), files.size());
        return records;
    }

    // ── Zillow listings JSON (data/zillow_seattle_listings.json format) ───────

    @SuppressWarnings("unchecked")
    public List<PropertyRecord> ingestZillowJson(File file) throws IOException {
        return ingestZillowJsonFiles(List.of(file));
    }

    /** Parse all saved Zillow responses, then score the de-duplicated source in one batch. */
    @SuppressWarnings("unchecked")
    public List<PropertyRecord> ingestZillowJsonFiles(List<File> files) throws IOException {
        LOG.infof("Ingesting %d Zillow JSON file(s) as one batch", files.size());
        long started = System.nanoTime();
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        List<PendingProperty> pending = new ArrayList<>();
        for (File file : files) {
            Map<String, Object> root;
            List<Map<String, Object>> properties;
            try {
                root = mapper.readValue(file,
                    mapper.getTypeFactory().constructMapType(Map.class, String.class, Object.class));
                properties = (List<Map<String, Object>>) root.get("properties");
            } catch (IOException exception) {
                LOG.warnf("Skipping unreadable Zillow JSON %s: %s", file.getName(), exception.getMessage());
                continue;
            }
            if (properties == null) continue;
            LocalDate observationDate = jsonObservationDate(root, file);
            for (Map<String, Object> p : properties) try {
                pending.add(zillowMapToPending(p, observationDate));
            } catch (Exception e) {
                LOG.debugf("Skipping malformed Zillow entry: %s", e.getMessage());
            }
        }
        long parsedAt = System.nanoTime();
        List<PropertyRecord> records = predictChanged(pending);
        LOG.infof("Ingested %d changed Zillow listings (%d parsed from %d file(s)) — parse %.1f ms, prediction pipeline %.1f ms, total %.1f ms",
            records.size(), pending.size(), files.size(), (parsedAt - started) / 1_000_000.0,
            (System.nanoTime() - parsedAt) / 1_000_000.0, (System.nanoTime() - started) / 1_000_000.0);
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
        List<PendingProperty> changed = unique.values().stream()
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
        boolean rentcast = !changed.isEmpty() && "rentcast".equals(changed.getFirst().source());
        long contextStarted = System.nanoTime();
        List<PredictionContext> contexts = rentcast
            ? contextFactory.prepareRentcastAll(inputs)
            : contextFactory.prepareAll(inputs, false);
        long contextElapsed = System.nanoTime() - contextStarted;
        long neuralStarted = System.nanoTime();
        EmbeddingModel.PredictionResult[] neural = model.predictPrepared(contexts);
        long neuralElapsed = System.nanoTime() - neuralStarted;
        long lightgbmStarted = System.nanoTime();
        double[] tree = lightgbm.predictPrepared(contexts);
        long lightgbmElapsed = System.nanoTime() - lightgbmStarted;
        long gnnStarted = System.nanoTime();
        double[] gnnPredictions = gnn.predictBatch(changed.stream().map(value -> new GnnModel.Input(
            value.h3Index(), value.saleDate(), value.sqft(), value.sqftLot(), value.beds(),
            value.lat(), value.lng()
        )).toList());
        long gnnElapsed = System.nanoTime() - gnnStarted;
        LOG.infof("Model prediction batch: %,d %s record(s) — context %.1f ms, neural %.1f ms, LightGBM %.1f ms, GNN %.1f ms",
            changed.size(), changed.getFirst().source(), contextElapsed / 1_000_000.0, neuralElapsed / 1_000_000.0,
            lightgbmElapsed / 1_000_000.0, gnnElapsed / 1_000_000.0);
        long recordBuildStarted = System.nanoTime();
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
            double gnnError = value.salePrice() > 0
                ? 100.0 * (gnnPredictions[index] - value.salePrice()) / value.salePrice()
                : 0.0;
            records.add(new PropertyRecord(
                value.id(), value.address(), value.source(), value.lat(), value.lng(),
                value.h3Index(), value.community(), value.sqft(), value.sqftLot(),
                value.beds(), value.baths(), value.homeType(), value.saleDate(),
                value.salePrice(), value.zestimate(), value.listingUrl(), prediction.predictedPrice(),
                neuralError, tree[index], treeError, gnnPredictions[index], gnnError, prediction.predictionStdPrice(),
                prediction.predictionCvPct(), prediction.clsAttention()
            ));
        }
        LOG.infof("Prediction record assembly: %,d %s record(s) — %.1f ms",
            records.size(), changed.getFirst().source(),
            (System.nanoTime() - recordBuildStarted) / 1_000_000.0);
        return records;
    }

    /** Score parsed historical rows in one lightweight GNN-head batch sequence. */
    private List<PropertyRecord> withGnnPredictions(List<PropertyRecord> records) {
        double[] predictions = gnn.predictBatch(records.stream().map(record -> new GnnModel.Input(
            record.h3Index(), record.saleDate(), record.sqft(), record.sqftLot(), record.beds(),
            record.lat(), record.lng()
        )).toList());
        List<PropertyRecord> result = new ArrayList<>(records.size());
        for (int index = 0; index < records.size(); index++) {
            PropertyRecord record = records.get(index);
            double error = record.salePrice() > 0
                ? 100.0 * (predictions[index] - record.salePrice()) / record.salePrice() : 0.0;
            result.add(new PropertyRecord(
                record.id(), record.address(), record.source(), record.lat(), record.lng(),
                record.h3Index(), record.community(), record.sqft(), record.sqftLot(), record.beds(),
                record.baths(), record.homeType(), record.saleDate(), record.salePrice(), record.zestimate(),
                record.listingUrl(), record.predictedPrice(), record.pctError(),
                record.lightgbmPredictedPrice(), record.lightgbmPctError(), predictions[index], error,
                record.predictionStdPrice(), record.predictionCvPct(), record.clsAttention()
            ));
        }
        return result;
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
            "cls_attn_community", "cls_attn_property",
            "cls_attn_time", "cls_attn_market"
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

    /**
     * Return the deterministic historical-sales identifier: numeric sale date
     * followed by the CSV row number. The internal store uses a String key so
     * the decimal value is never rounded by a JavaScript or floating type.
     */
    static String buildUniqueRecordId(LocalDate saleDate, int rowNumber) {
        String saleDateDigits = saleDate.toString().replaceAll("\\D", "");
        return new java.math.BigInteger(saleDateDigits + rowNumber).toString();
    }

    private LocalDate parseDateOrDefault(Object o) {
        try { return parseLocalDate(String.valueOf(o)); }
        catch (Exception e) { return LocalDate.now(); }
    }

}
