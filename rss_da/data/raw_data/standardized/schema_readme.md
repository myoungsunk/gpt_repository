
# rss_da Standardized Dataset Schema

## Common fields per sample
- sample_id: int
- split: str ("train"/"val"/"calib"/"test")  [default via CLI]
- domain_id: int  (0=Stage-1, 1=Stage-2.5)
- is_labeled: int (1/0)

### z5d (5 features, dBm-based)
- rss1_combined_dbm: float
- rss2_combined_dbm: float
- delta_db: float  (rss2 - rss1 in dB)
- sum_db: float    (rss1 + rss2 in dB)
- log_ratio: float (log(P2_lin/P1_lin), P in mW)

### c_meas_dbm (duplicated from z5d[0:2])
- c1_dbm, c2_dbm

### four_rss_dbm (Stage-1 only; NaN in Stage-2.5)
- rss_a_p1_dbm, rss_b_p1_dbm, rss_a_p2_dbm, rss_b_p2_dbm

### angles
- theta1_rad, theta2_rad

### geometry
- d1_m, d2_m, room_dim_m

### flags
- mask_4rss_is_gt: 1 if GT(only Stage-1), else 0

## Physics LUT parquet (single_los_lut.parquet)
- distance_m: float
- theta_deg: float
- theta_rad: float
- rss1_db: float
- rss2_db: float
