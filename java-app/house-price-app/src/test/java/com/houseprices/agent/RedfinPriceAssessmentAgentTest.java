package com.houseprices.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RedfinPriceAssessmentAgentTest {

    @Test
    void keepsSystemAndUserPromptTemplatesSeparate() throws Exception {
        Method assess = RedfinPriceAssessmentAgent.PriceAnalyst.class.getDeclaredMethod(
            "assess", String.class, String.class);
        SystemMessage systemMessage = assess.getAnnotation(SystemMessage.class);

        assertNotNull(systemMessage);
        assertNotNull(assess.getAnnotation(UserMessage.class));
        String prompt = String.join("\n", systemMessage.value());
        assertFalse(prompt.contains("@UserMessage"));
        assertTrue(prompt.contains("Return only a concise semantic HTML fragment"));
        assertTrue(prompt.contains("href is exactly the supplied listingUrl"));
        assertFalse(prompt.contains("without html,\n            head, body, style, script, links"));
        assertTrue(prompt.contains("End the response with an <h3>Assessment rationale</h3>"));
        assertTrue(prompt.indexOf("Assessment rationale") > prompt.indexOf("Confidence and limitations"));
    }

    @Test
    void browserPresentationSanitizesAgentHtmlBeforeRendering() throws Exception {
        try (var input = getClass().getClassLoader().getResourceAsStream(
                "META-INF/resources/js/agent.js")) {
            assertNotNull(input);
            String javascript = new String(input.readAllBytes(), StandardCharsets.UTF_8);
            assertTrue(javascript.contains("sanitizeAgentHtml(payload.assessment)"));
            assertTrue(javascript.contains("safeAgentListingUrl(element.getAttribute('href'))"));
            assertTrue(javascript.contains("url.protocol !== 'https:'"));
            assertTrue(javascript.contains("hostname.endsWith('.redfin.com')"));
            assertTrue(javascript.contains("element.setAttribute('rel', 'noopener noreferrer')"));
            assertTrue(javascript.contains("element.removeAttribute(attribute.name)"));
            assertFalse(javascript.contains("escapeHtml(payload.assessment)"));
        }
    }
}
