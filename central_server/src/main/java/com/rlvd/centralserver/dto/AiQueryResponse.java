package com.rlvd.centralserver.dto;

import java.util.List;
import java.util.Map;

/**
 * Response DTO from the AI Text-to-SQL pipeline.
 * Contains the original question, generated SQL, query results, and viz hint.
 */
public class AiQueryResponse {

    private String question;
    private String sql;
    private String answer;
    private List<String> columns;
    private List<Map<String, Object>> rows;
    private int count;
    private String chartType; // "table" or "bar"
    private String error;

    public AiQueryResponse() {}

    public String getQuestion() { return question; }
    public void setQuestion(String question) { this.question = question; }

    public String getSql() { return sql; }
    public void setSql(String sql) { this.sql = sql; }

    public String getAnswer() { return answer; }
    public void setAnswer(String answer) { this.answer = answer; }

    public List<String> getColumns() { return columns; }
    public void setColumns(List<String> columns) { this.columns = columns; }

    public List<Map<String, Object>> getRows() { return rows; }
    public void setRows(List<Map<String, Object>> rows) { this.rows = rows; }

    public int getCount() { return count; }
    public void setCount(int count) { this.count = count; }

    public String getChartType() { return chartType; }
    public void setChartType(String chartType) { this.chartType = chartType; }

    public String getError() { return error; }
    public void setError(String error) { this.error = error; }
}