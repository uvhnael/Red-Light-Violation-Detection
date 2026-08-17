package com.rlvd.centralserver.dto;

import lombok.*;

/**
 * License plate OCR result.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PlateDto {
    private String text;
    private Double confidence;
}
