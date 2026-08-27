package com.rlvd.centralserver.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.util.List;

/**
 * Paged violation list response — the frontend loads one page at a time
 * instead of pulling the whole table.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ViolationPageResponse {

    private List<ViolationResponse> content;

    private int page;

    private int size;

    @JsonProperty("total_elements")
    private long totalElements;

    @JsonProperty("total_pages")
    private int totalPages;

    private boolean first;

    private boolean last;
}
