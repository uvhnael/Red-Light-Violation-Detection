package com.rlvd.centralserver.dto;

import java.util.List;
import java.util.Map;

/**
 * Request DTO for AI natural language query.
 */
public class AiQueryRequest {

    private String question;

    public AiQueryRequest() {}

    public AiQueryRequest(String question) {
        this.question = question;
    }

    public String getQuestion() { return question; }
    public void setQuestion(String question) { this.question = question; }
}