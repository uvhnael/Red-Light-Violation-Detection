package com.rlvd.centralserver.dto;

import lombok.*;

/**
 * Represents a 2D point with x/y coordinates.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PointDto {
    private Double x;
    private Double y;
}
