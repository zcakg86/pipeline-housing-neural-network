package com.houseprices.model;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

/** Causal, file-backed lookup for the market indicators used at inference. */
@ApplicationScoped
public class MarketIndicatorService {

    private static final Logger LOG = Logger.getLogger(MarketIndicatorService.class);
    private static final String RESOURCE = "model-artifacts/fred_indicators.csv";

    public record MarketIndicators(
        LocalDate effectiveDate,
        double mortgageRate,
        double unemploymentRate
    ) {}

    private NavigableMap<LocalDate, MarketIndicators> indicators;

    @PostConstruct
    void load() {
        NavigableMap<LocalDate, MarketIndicators> loaded = new TreeMap<>();
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(RESOURCE)) {
            if (input == null) {
                throw new IllegalStateException(RESOURCE + " not found in resources");
            }
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(input, StandardCharsets.UTF_8))) {
                String header = reader.readLine();
                if (!"date,mortgage_rate,unemployment_rate".equals(header)) {
                    throw new IllegalStateException("Unexpected market-indicator CSV header");
                }
                String line;
                int lineNumber = 1;
                while ((line = reader.readLine()) != null) {
                    lineNumber++;
                    if (line.isBlank()) continue;
                    String[] fields = line.split(",", -1);
                    if (fields.length != 3) {
                        throw new IllegalStateException(
                            "Malformed market-indicator row at line " + lineNumber
                        );
                    }
                    LocalDate date = LocalDate.parse(fields[0]);
                    double mortgageRate = Double.parseDouble(fields[1]);
                    double unemploymentRate = Double.parseDouble(fields[2]);
                    if (!Double.isFinite(mortgageRate) || !Double.isFinite(unemploymentRate)) {
                        throw new IllegalStateException(
                            "Non-finite market indicator at line " + lineNumber
                        );
                    }
                    loaded.put(
                        date,
                        new MarketIndicators(date, mortgageRate, unemploymentRate)
                    );
                }
            }
        } catch (Exception exception) {
            throw new RuntimeException("Failed to load market indicators", exception);
        }
        if (loaded.isEmpty()) {
            throw new IllegalStateException("Market-indicator history is empty");
        }
        indicators = loaded;
        LOG.infof(
            "Loaded %,d daily market-indicator rows (%s to %s)",
            indicators.size(), indicators.firstKey(), indicators.lastKey()
        );
    }

    /** Return the latest values dated on or before the prediction date. */
    public MarketIndicators lookup(LocalDate predictionDate) {
        if (predictionDate == null) {
            throw new IllegalArgumentException("Prediction date is required");
        }
        Map.Entry<LocalDate, MarketIndicators> match = indicators.floorEntry(predictionDate);
        if (match == null) {
            throw new IllegalArgumentException(
                "No market indicators are available on or before " + predictionDate
            );
        }
        return match.getValue();
    }
}
