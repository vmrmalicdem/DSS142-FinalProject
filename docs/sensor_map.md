# Sensor-to-Room Mapping

Source: Candanedo, L.M. & Feldheim, V. (2016). "Predicting home appliance energy use: A statistical analysis of the role of occupancy and weather on energy consumption." *Energy and Buildings*, 116, 310–321. UCI Machine Learning Repository ID: 374.

> **⚠ WARNING**: **T6 and RH_6 are outdoor sensors**, not room sensors. They are included in the raw dataset's column list but must be excluded from any "rooms" analysis. In Notebook 2's Q2 analysis, `Q_VARS` currently includes these outdoor sensors. Results for T6/RH_6 from Q2 should be interpreted as outdoor environment associations, not room threshold crossings.

## Temperature Sensors

| Sensor | Location | Type | Notes |
|--------|----------|------|-------|
| T1 | Kitchen | Indoor | — |
| T2 | Living room | Indoor | — |
| T3 | Laundry room | Indoor | — |
| T4 | Office room | Indoor | — |
| T5 | Bathroom | Indoor | — |
| T6 | Outside building (north) | **Outdoor** | ⚠ Outdoor — exclude from room-based analysis |
| T7 | Ironing room | Indoor | — |
| T8 | Teenager bedroom | Indoor | — |
| T9 | Parents' bedroom | Indoor | — |
| T_out | Outside temperature | **Outdoor** | From Chievres Weather Station |

## Humidity Sensors

| Sensor | Location | Type | Notes |
|--------|----------|------|-------|
| RH_1 | Kitchen | Indoor | — |
| RH_2 | Living room | Indoor | — |
| RH_3 | Laundry room | Indoor | — |
| RH_4 | Office room | Indoor | — |
| RH_5 | Bathroom | Indoor | Flagged for spikes via Tukey fence (not clipped) |
| RH_6 | Outside building (north) | **Outdoor** | ⚠ Outdoor — exclude from room-based analysis; used in Q2 sensitivity (S1) |
| RH_7 | Ironing room | Indoor | — |
| RH_8 | Teenager bedroom | Indoor | — |
| RH_9 | Parents' bedroom | Indoor | — |
| RH_out | Outside humidity | **Outdoor** | From Chievres Weather Station |

## Feature Group Assignments (as used in Q3)

| Group name | Sensors included |
|-----------|-----------------|
| `indoor` | T1–T5, T7–T9, RH_1–RH_5, RH_7–RH_9 |
| `outdoor` | T6, T_out, RH_6, RH_out, Windspeed, Visibility, Tdewpoint |
| `indoor_outdoor` | All of `indoor` + `outdoor` |
| `temporal` | hour, day_of_week (engineered features) |
| `combined` | All features (`indoor_outdoor` + `temporal`) |

## Audit Notes (G10)

- **Q2 concern**: `Q_VARS` in NB2 includes `T6` and `RH_6`. Their threshold test results reflect outdoor conditions, not room-level thresholds. Label them clearly in any report table.
- **Q3 grouping is correct**: The `outdoor` group correctly contains `T6` and `RH_6` as outdoor sensors.
