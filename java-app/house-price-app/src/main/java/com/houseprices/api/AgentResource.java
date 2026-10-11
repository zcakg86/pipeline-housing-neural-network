package com.houseprices.api;

import com.houseprices.agent.AgentEvaluationStore;
import com.houseprices.agent.RedfinPriceAssessmentAgent;
import jakarta.inject.Inject;
import jakarta.ws.rs.BadRequestException;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.InternalServerErrorException;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.util.Map;

/** HTTP boundary for price assessments and human evaluation feedback. */
@Path("/api/agent")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class AgentResource {

    private static final Logger LOG = Logger.getLogger(AgentResource.class);

    @Inject RedfinPriceAssessmentAgent agent;
    @Inject AgentEvaluationStore evaluations;

    @POST
    @Path("/redfin/{listingId}/assessment")
    public Response assess(
            @PathParam("listingId") String listingId, AssessmentRequest request) {
        if (listingId == null || listingId.isBlank()) {
            return error(Response.Status.BAD_REQUEST, "listing_id_required", "listingId is required");
        }
        try {
            return Response.ok(agent.assess(
                listingId, request == null ? null : request.question()
            )).build();
        } catch (IllegalArgumentException exception) {
            LOG.warnf(exception, "Agent evidence could not be built for Redfin listing %s", listingId);
            return error(Response.Status.NOT_FOUND, "listing_or_evidence_not_found", message(exception));
        } catch (IllegalStateException exception) {
            LOG.warnf(exception, "Agent is unavailable for Redfin listing %s", listingId);
            return error(Response.Status.SERVICE_UNAVAILABLE, "agent_unavailable", message(exception));
        } catch (Exception exception) {
            LOG.errorf(exception, "Agent assessment failed for Redfin listing %s", listingId);
            return error(Response.Status.BAD_GATEWAY, "assessment_failed", providerMessage(exception));
        }
    }

    @POST
    @Path("/assessments/{assessmentId}/feedback")
    public Map<String, Object> feedback(
            @PathParam("assessmentId") String assessmentId,
            AgentEvaluationStore.Feedback feedback) {
        if (feedback == null) throw new BadRequestException("feedback body is required");
        try {
            return evaluations.saveFeedback(assessmentId, feedback);
        } catch (IllegalArgumentException exception) {
            throw new BadRequestException(exception.getMessage());
        } catch (IOException exception) {
            throw new InternalServerErrorException("Could not save feedback", exception);
        }
    }

    @GET
    @Path("/evaluations/summary")
    public Map<String, Object> evaluationSummary() {
        try { return evaluations.summary(); }
        catch (IOException exception) {
            throw new InternalServerErrorException("Could not read agent evaluations", exception);
        }
    }

    private Response error(Response.Status status, String code, String detail) {
        return Response.status(status).entity(Map.of(
            "error", code,
            "detail", detail,
            "status", status.getStatusCode()
        )).build();
    }

    private String message(Exception exception) {
        String detail = exception.getMessage();
        if (detail == null || detail.isBlank()) return exception.getClass().getSimpleName();
        return detail;
    }

    private String providerMessage(Exception exception) {
        String detail = message(exception);
        String normalized = detail.toLowerCase();
        if (normalized.contains("no remaining credits") || normalized.contains("status code: 402")
                || normalized.contains("status=402")) {
            return "Hugging Face Inference Providers returned HTTP 402: this account has no "
                + "remaining inference credits. Add prepaid credits, use included PRO credits, "
                + "or configure another funded provider/model, then retry.";
        }
        if (normalized.contains("status code: 401") || normalized.contains("status=401")) {
            return "Hugging Face rejected HF_TOKEN (HTTP 401). Check the token and its "
                + "Inference Providers permission, then restart the app.";
        }
        if (normalized.contains("status code: 429") || normalized.contains("status=429")) {
            return "Hugging Face rate-limited the request (HTTP 429). Wait and retry, or select "
                + "a provider/model with available capacity.";
        }
        return detail;
    }

    public record AssessmentRequest(String question) {}
}
