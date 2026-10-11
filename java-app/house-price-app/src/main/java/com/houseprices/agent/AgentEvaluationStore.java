package com.houseprices.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Durable, append-only assessment and human-feedback artifacts for evaluation. */
@ApplicationScoped
public class AgentEvaluationStore {

    @Inject ObjectMapper mapper;

    @ConfigProperty(name = "agent.evaluation.dir", defaultValue = "data/agent-evaluations")
    String evaluationDir;

    public synchronized void saveAssessment(Map<String, Object> assessment) throws IOException {
        String id = requiredId(assessment.get("assessmentId"));
        writeAtomically(directory("assessments").resolve(id + ".json"), assessment);
    }

    public synchronized Map<String, Object> saveFeedback(String assessmentId, Feedback feedback)
            throws IOException {
        String safeId = requiredId(assessmentId);
        Path assessment = directory("assessments").resolve(safeId + ".json");
        if (!Files.isRegularFile(assessment)) {
            throw new IllegalArgumentException("Assessment not found: " + safeId);
        }
        if (feedback.rating() < 1 || feedback.rating() > 5) {
            throw new IllegalArgumentException("rating must be between 1 and 5");
        }
        if (feedback.expectedPrice() != null && feedback.expectedPrice() < 0) {
            throw new IllegalArgumentException("expectedPrice cannot be negative");
        }
        Map<String, Object> record = new LinkedHashMap<>();
        record.put("feedbackId", UUID.randomUUID().toString());
        record.put("assessmentId", safeId);
        record.put("createdAt", Instant.now().toString());
        record.put("rating", feedback.rating());
        record.put("expectedPrice", feedback.expectedPrice());
        record.put("comments", feedback.comments() == null ? "" : feedback.comments().trim());
        String filename = record.get("createdAt").toString().replace(':', '-') + "_"
            + record.get("feedbackId") + ".json";
        writeAtomically(directory("feedback").resolve(filename), record);
        return record;
    }

    public synchronized Map<String, Object> summary() throws IOException {
        Path feedbackDirectory = directory("feedback");
        List<Map<String, Object>> feedback = new ArrayList<>();
        try (var paths = Files.list(feedbackDirectory)) {
            for (Path path : paths.filter(value -> value.toString().endsWith(".json")).toList()) {
                feedback.add(mapper.readValue(path.toFile(), Map.class));
            }
        }
        double average = feedback.stream()
            .map(Map.class::cast)
            .map(value -> value.get("rating"))
            .filter(Number.class::isInstance)
            .map(Number.class::cast)
            .mapToDouble(Number::doubleValue)
            .average().orElse(0);
        long assessments;
        try (var paths = Files.list(directory("assessments"))) {
            assessments = paths.filter(value -> value.toString().endsWith(".json")).count();
        }
        return Map.of(
            "assessmentCount", assessments,
            "feedbackCount", feedback.size(),
            "averageRating", average,
            "feedback", feedback
        );
    }

    private Path directory(String child) throws IOException {
        Path path = Path.of(evaluationDir).resolve(child);
        Files.createDirectories(path);
        return path;
    }

    private String requiredId(Object value) {
        String text = String.valueOf(value);
        try { return UUID.fromString(text).toString(); }
        catch (Exception exception) { throw new IllegalArgumentException("Invalid assessment ID"); }
    }

    private void writeAtomically(Path target, Object value) throws IOException {
        Path temporary = Files.createTempFile(target.getParent(), target.getFileName().toString(), ".tmp");
        try {
            mapper.writerWithDefaultPrettyPrinter().writeValue(temporary.toFile(), value);
            try {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException exception) {
                Files.move(temporary, target);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public record Feedback(int rating, Double expectedPrice, String comments) {}
}
