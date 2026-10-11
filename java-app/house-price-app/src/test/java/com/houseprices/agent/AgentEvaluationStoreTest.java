package com.houseprices.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentEvaluationStoreTest {

    @TempDir Path temporaryDirectory;

    @Test
    void storesAssessmentsAndAppendOnlyFeedbackForEvaluation() throws Exception {
        AgentEvaluationStore store = new AgentEvaluationStore();
        store.mapper = new ObjectMapper();
        store.evaluationDir = temporaryDirectory.toString();
        String assessmentId = UUID.randomUUID().toString();

        store.saveAssessment(Map.of(
            "assessmentId", assessmentId,
            "assessment", "Evidence-grounded result",
            "evidence", Map.of("listing", Map.of("id", "redfin-1"))
        ));
        Map<String, Object> first = store.saveFeedback(
            assessmentId, new AgentEvaluationStore.Feedback(4, 825_000.0, "Useful")
        );
        Map<String, Object> second = store.saveFeedback(
            assessmentId, new AgentEvaluationStore.Feedback(2, null, "Missed a risk")
        );

        assertTrue(!first.get("feedbackId").equals(second.get("feedbackId")));
        Map<String, Object> summary = store.summary();
        assertEquals(1L, summary.get("assessmentCount"));
        assertEquals(2, summary.get("feedbackCount"));
        assertEquals(3.0, (double) summary.get("averageRating"), 0.0001);
    }
}
