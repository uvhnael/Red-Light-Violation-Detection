package com.rlvd.centralserver.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.util.List;

/**
 * Response for a batch ingest call. Reports how many violations were
 * accepted vs. skipped as duplicates so the edge outbox can mark rows sent.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ViolationBatchResponse {

    private int accepted;

    private int duplicates;

    private int failed;

    @JsonProperty("accepted_event_ids")
    private List<String> acceptedEventIds;

    @JsonProperty("duplicate_event_ids")
    private List<String> duplicateEventIds;
}
