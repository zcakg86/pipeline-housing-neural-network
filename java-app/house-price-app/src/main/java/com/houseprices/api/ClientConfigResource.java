package com.houseprices.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.Map;
import java.util.Optional;

/** Exposes the browser-safe runtime configuration needed by the static UI. */
@Path("/api/client-config.js")
public class ClientConfigResource {

    @Inject ObjectMapper mapper;

    @ConfigProperty(name = "carto.api.key")
    Optional<String> cartoApiKey;

    @GET
    @Produces("application/javascript")
    public Response clientConfig() throws JsonProcessingException {
        String json = mapper.writeValueAsString(Map.of(
            "cartoApiKey", cartoApiKey.orElse("")
        ));
        return Response.ok("window.HOUSE_PRICE_CONFIG = Object.freeze(" + json + ");\n")
            .header("Cache-Control", "no-store")
            .build();
    }
}
