package com.houseprices.ingest;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RedfinApiClientTest {

    @Test
    void handlesEachSuccessBeforeStoppingOnALaterDenial() throws Exception {
        List<String> events = new ArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/redfin", exchange -> {
            String query = exchange.getRequestURI().getRawQuery();
            String keyword = query.contains("keyword=98144") ? "98144" : "98004";
            events.add("request-" + keyword);
            if ("98004".equals(keyword)) {
                exchange.sendResponseHeaders(429, -1);
            } else {
                byte[] bytes = "{\"properties\":[{\"id\":1}]}"
                    .getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            }
            exchange.close();
        });
        server.start();
        try {
            RedfinApiClient client = configuredClient(server);
            client.keywords = "98144,98004,98005";

            RedfinApiClient.FetchResult result = client.fetchListings(false,
                (keyword, page, response) -> events.add("saved-" + keyword));

            assertEquals(List.of("request-98144", "saved-98144", "request-98004"), events);
            assertEquals(1, result.pageResponses().size());
            assertEquals(429, result.failure().statusCode());
            assertEquals("98004", result.failure().keyword());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void fetchesOnePageForEveryConfiguredKeywordByDefault() throws Exception {
        List<String> queries = new ArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/redfin", exchange -> {
            queries.add(exchange.getRequestURI().getRawQuery());
            String body = "{\"properties\":[],\"pagination\":{\"currentPage\":1}}";
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        try {
            RedfinApiClient client = configuredClient(server);
            client.keywords = "98144, 98004";

            RedfinApiClient.FetchResult result = client.fetchListings(false);

            assertEquals(2, result.pageResponses().size());
            assertTrue(queries.getFirst().contains("keyword=98144"));
            assertTrue(queries.getLast().contains("keyword=98004"));
            assertTrue(queries.stream().noneMatch(query -> query.contains("page=2")));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void followsResponsePaginationAndPreservesCompletePages() throws Exception {
        List<String> queries = new ArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/redfin", exchange -> {
            String query = exchange.getRequestURI().getRawQuery();
            queries.add(query);
            boolean second = query != null && query.contains("page=2");
            String body = second
                ? """
                    {"requestMetadata":{"id":"page-2"},"properties":[
                      {"id":2,"description":"Second description","atAGlanceFacts":["Pool"]}
                    ],"pagination":{"currentPage":2}}
                    """
                : """
                    {"requestMetadata":{"id":"page-1"},"searchInformation":{"totalResults":2},
                     "properties":[{"id":1,"description":"First description",
                     "atAGlanceFacts":["Open kitchen"]}],
                     "pagination":{"currentPage":1,
                     "nextPage":"https://redfin.com/zipcode/98144/page-2"}}
                    """;
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        try {
            RedfinApiClient client = configuredClient(server);
            client.keywords = "98144";

            RedfinApiClient.FetchResult result = client.fetchListings(true);

            assertEquals(2, result.properties().size());
            assertEquals(2, result.pageResponses().size());
            assertEquals("page-1", ((java.util.Map<?, ?>)
                result.pageResponses().getFirst().get("requestMetadata")).get("id"));
            assertEquals(2, ((List<?>) result.savedResponse().get("pages")).size());
            assertEquals("Second description", result.properties().getLast().get("description"));
            assertTrue(queries.getFirst().contains("keyword=98144"));
            assertTrue(queries.getFirst().contains("homeTypes%5B%5D=condo"));
            assertTrue(queries.getFirst().contains("sort=newest"));
            assertTrue(!queries.getFirst().contains("timeOnRedfin"));
            assertTrue(queries.getLast().contains("page=2"));
        } finally {
            server.stop(0);
        }
    }

    private RedfinApiClient configuredClient(HttpServer server) {
        RedfinApiClient client = new RedfinApiClient();
        client.mapper = new ObjectMapper();
        client.apiUrl = "http://localhost:" + server.getAddress().getPort() + "/redfin";
        client.apiKey = "test-key";
        client.listingType = "forSale";
        client.sort = "newest";
        client.homeTypes = "condo,townhouse,house";
        client.timeOnRedfin = java.util.Optional.empty();
        client.maxPages = 10;
        client.delayMs = 0;
        return client;
    }
}
