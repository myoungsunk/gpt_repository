
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_datasets_v2.py
- Supports attenuation mode for linear inputs (unitless a_lin), with optional Pt(dBm)
- Generates relative-dB fields (c*_rel_db, four_rss_rel_db*), and optionally absolute dBm if Pt provided
- Adds CSV fallback for physics LUT if pyarrow/fastparquet is unavailable
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

EPS = 1e-20

def lin_to_db_ratio(x):
    """Convert linear ratio (unitless) to dB (relative)."""
    x = np.asarray(x, dtype=np.float64)
    return 10.0 * np.log10(np.clip(x, EPS, None))

def db_ratio_to_lin(x_db):
    """Convert dB (relative) to linear ratio (unitless)."""
    x_db = np.asarray(x_db, dtype=np.float64)
    return np.power(10.0, x_db / 10.0)

def linear_power_to_dbm(p_lin, linear_unit="w"):
    """Absolute power to dBm (W or mW)."""
    p_lin = np.asarray(p_lin, dtype=np.float64)
    p_safe = np.clip(p_lin, EPS, None)
    if linear_unit.lower() == "w":
        return 10.0 * np.log10(p_safe * 1e3)  # W -> mW
    elif linear_unit.lower() == "mw":
        return 10.0 * np.log10(p_safe)
    else:
        raise ValueError("linear_unit must be 'w' or 'mw'")

def build_z5d_from_db(a_db, b_db):
    """Given two dB-scalars (relative or absolute), build delta/sum/log_ratio (using linear domain for ratio)."""
    a_db = np.asarray(a_db, dtype=np.float64)
    b_db = np.asarray(b_db, dtype=np.float64)
    delta_db = b_db - a_db
    sum_db = a_db + b_db
    Pa = db_ratio_to_lin(a_db)  # if absolute dBm, 'ratio' is still fine up to a constant; prefer relative dB
    Pb = db_ratio_to_lin(b_db)
    log_ratio = np.log(np.clip(Pb / np.clip(Pa, EPS, None), EPS, None))
    return delta_db, sum_db, log_ratio

def convert_stage1(path_in, path_out, mode="atten", pt_dbm=None, split="train", domain_id=0):
    df = pd.read_csv(path_in)
    req = ["RSS1_LoS1","RSS2_LoS1","RSS1_LoS2","RSS2_LoS2","RSS1_combined","RSS2_combined","d1","d2","room_dim_meter","theta1_true","theta2_true"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Stage-1 CSV missing columns: {missing}")

    if mode == "atten":
        c1_rel_db = lin_to_db_ratio(df["RSS1_combined"].values)
        c2_rel_db = lin_to_db_ratio(df["RSS2_combined"].values)
        r_a_p1_rel_db = lin_to_db_ratio(df["RSS1_LoS1"].values)
        r_b_p1_rel_db = lin_to_db_ratio(df["RSS2_LoS1"].values)
        r_a_p2_rel_db = lin_to_db_ratio(df["RSS1_LoS2"].values)
        r_b_p2_rel_db = lin_to_db_ratio(df["RSS2_LoS2"].values)
        # Optional absolute dBm (if Pt provided)
        if pt_dbm is not None:
            c1_dbm = c1_rel_db + pt_dbm
            c2_dbm = c2_rel_db + pt_dbm
        else:
            c1_dbm = np.full_like(c1_rel_db, np.nan)
            c2_dbm = np.full_like(c2_rel_db, np.nan)
    elif mode in ("w","mw"):
        # Treat as absolute powers
        c1_dbm = linear_power_to_dbm(df["RSS1_combined"].values, mode)
        c2_dbm = linear_power_to_dbm(df["RSS2_combined"].values, mode)
        r_a_p1_rel_db = lin_to_db_ratio(df["RSS1_LoS1"].values / np.maximum(df["RSS1_combined"].values, EPS))  # placeholder
        r_b_p1_rel_db = lin_to_db_ratio(df["RSS2_LoS1"].values / np.maximum(df["RSS2_combined"].values, EPS))
        r_a_p2_rel_db = lin_to_db_ratio(df["RSS1_LoS2"].values / np.maximum(df["RSS1_combined"].values, EPS))
        r_b_p2_rel_db = lin_to_db_ratio(df["RSS2_LoS2"].values / np.maximum(df["RSS2_combined"].values, EPS))
        c1_rel_db = c1_dbm - np.nanmean(c1_dbm)
        c2_rel_db = c2_dbm - np.nanmean(c2_dbm)
    else:
        raise ValueError("mode must be 'atten', 'w', or 'mw'")

    delta_rel_db, sum_rel_db, log_ratio = build_z5d_from_db(c1_rel_db, c2_rel_db)

    out = pd.DataFrame({
        "sample_id": np.arange(len(df)),
        "split": split,
        "domain_id": domain_id,
        "is_labeled": 1,
        # z5d (relative dB canonical)
        "c1_rel_db": c1_rel_db,
        "c2_rel_db": c2_rel_db,
        "delta_rel_db": delta_rel_db,
        "sum_rel_db": sum_rel_db,
        "log_ratio": log_ratio,
        # (optional) absolute dBm
        "c1_dbm": c1_dbm,
        "c2_dbm": c2_dbm,
        # four RSS (relative dB)
        "rss_a_p1_rel_db": r_a_p1_rel_db,
        "rss_b_p1_rel_db": r_b_p1_rel_db,
        "rss_a_p2_rel_db": r_a_p2_rel_db,
        "rss_b_p2_rel_db": r_b_p2_rel_db,
        # angles & geometry
        "theta1_rad": df["theta1_true"].values.astype(np.float64),
        "theta2_rad": df["theta2_true"].values.astype(np.float64),
        "d1_m": df["d1"].values.astype(np.float64),
        "d2_m": df["d2"].values.astype(np.float64),
        "room_dim_m": df["room_dim_meter"].values.astype(np.float64),
        # flags
        "mask_4rss_is_gt": 1,
    })

    cols = [
        "sample_id","split","domain_id","is_labeled",
        "c1_rel_db","c2_rel_db","delta_rel_db","sum_rel_db","log_ratio",
        "c1_dbm","c2_dbm",
        "rss_a_p1_rel_db","rss_b_p1_rel_db","rss_a_p2_rel_db","rss_b_p2_rel_db",
        "theta1_rad","theta2_rad","d1_m","d2_m","room_dim_m","mask_4rss_is_gt"
    ]
    out = out[cols]
    out.to_csv(path_out, index=False)
    return out

def convert_stage2(path_in, path_out, mode="atten", pt_dbm=None, split="train", domain_id=1):
    df = pd.read_csv(path_in)
    req = ["room_dim_meter","theta1_true","theta2_true","RSS1_combined","RSS2_combined","d1","d2"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Stage-2 CSV missing columns: {missing}")

    if mode == "atten":
        c1_rel_db = lin_to_db_ratio(df["RSS1_combined"].values)
        c2_rel_db = lin_to_db_ratio(df["RSS2_combined"].values)
        if pt_dbm is not None:
            c1_dbm = c1_rel_db + pt_dbm
            c2_dbm = c2_rel_db + pt_dbm
        else:
            c1_dbm = np.full_like(c1_rel_db, np.nan)
            c2_dbm = np.full_like(c2_rel_db, np.nan)
    elif mode in ("w","mw"):
        c1_dbm = linear_power_to_dbm(df["RSS1_combined"].values, mode)
        c2_dbm = linear_power_to_dbm(df["RSS2_combined"].values, mode)
        c1_rel_db = c1_dbm - np.nanmean(c1_dbm)
        c2_rel_db = c2_dbm - np.nanmean(c2_dbm)
    else:
        raise ValueError("mode must be 'atten', 'w', or 'mw'")

    delta_rel_db, sum_rel_db, log_ratio = build_z5d_from_db(c1_rel_db, c2_rel_db)

    out = pd.DataFrame({
        "sample_id": np.arange(len(df)),
        "split": split,
        "domain_id": domain_id,
        "is_labeled": 1,
        "c1_rel_db": c1_rel_db,
        "c2_rel_db": c2_rel_db,
        "delta_rel_db": delta_rel_db,
        "sum_rel_db": sum_rel_db,
        "log_ratio": log_ratio,
        "c1_dbm": c1_dbm,
        "c2_dbm": c2_dbm,
        "rss_a_p1_rel_db": np.nan,
        "rss_b_p1_rel_db": np.nan,
        "rss_a_p2_rel_db": np.nan,
        "rss_b_p2_rel_db": np.nan,
        "theta1_rad": df["theta1_true"].values.astype(np.float64),
        "theta2_rad": df["theta2_true"].values.astype(np.float64),
        "d1_m": df["d1"].values.astype(np.float64),
        "d2_m": df["d2"].values.astype(np.float64),
        "room_dim_m": df["room_dim_meter"].values.astype(np.float64),
        "mask_4rss_is_gt": 0,
    })
    cols = [
        "sample_id","split","domain_id","is_labeled",
        "c1_rel_db","c2_rel_db","delta_rel_db","sum_rel_db","log_ratio",
        "c1_dbm","c2_dbm",
        "rss_a_p1_rel_db","rss_b_p1_rel_db","rss_a_p2_rel_db","rss_b_p2_rel_db",
        "theta1_rad","theta2_rad","d1_m","d2_m","room_dim_m","mask_4rss_is_gt"
    ]
    out = out[cols]
    out.to_csv(path_out, index=False)
    return out

def convert_physics_csv(path_in, out_parquet):
    df = pd.read_csv(path_in)
    req = ["room_dim_meter","theta_true","RSS1_dB","RSS2_dB"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Physics CSV missing columns: {missing}")
    df = df.copy()
    df["theta_deg"] = df["theta_true"].astype(float)
    df["theta_rad"] = np.deg2rad(df["theta_deg"].values)
    df = df.rename(columns={"room_dim_meter":"distance_m","RSS1_dB":"rss1_db","RSS2_dB":"rss2_db"})
    subset = df[["distance_m","theta_deg","theta_rad","rss1_db","rss2_db"]]
    # try parquet, fallback to csv
    try:
        subset.to_parquet(out_parquet, index=False)
        return str(out_parquet)
    except Exception:
        csv_fallback = str(Path(out_parquet).with_suffix(".csv"))
        subset.to_csv(csv_fallback, index=False)
        return csv_fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_csv", type=str, default="module_input_sim_m15.csv")
    ap.add_argument("--stage2_csv", type=str, default="module_input_sim_nlos.csv")
    ap.add_argument("--physics_csv", type=str, default="single_los_rss_data.csv")
    ap.add_argument("--mode", type=str, default="atten", choices=["atten","w","mw"],
                    help="Interpretation of linear columns: attenuation ratio (unitless), Watts, or mW")
    ap.add_argument("--pt_dbm", type=float, default=None,
                    help="If provided, also output absolute dBm columns as rel_db + pt_dbm")
    ap.add_argument("--out_dir", type=str, default="standardized_v2")
    ap.add_argument("--s1_split", type=str, default="train")
    ap.add_argument("--s2_split", type=str, default="train")
    ap.add_argument("--s1_domain_id", type=int, default=0)
    ap.add_argument("--s2_domain_id", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_out = out_dir / "stage1_rel.csv"
    s2_out = out_dir / "stage25_rel.csv"
    phys_out = out_dir / "single_los_lut.parquet"

    df1 = convert_stage1(args.stage1_csv, str(s1_out),
                         mode=args.mode, pt_dbm=args.pt_dbm,
                         split=args.s1_split, domain_id=args.s1_domain_id)
    print(f"[OK] Stage-1(rel) → {s1_out} (N={len(df1)})")

    df2 = convert_stage2(args.stage2_csv, str(s2_out),
                         mode=args.mode, pt_dbm=args.pt_dbm,
                         split=args.s2_split, domain_id=args.s2_domain_id)
    print(f"[OK] Stage-2.5(rel) → {s2_out} (N={len(df2)})")

    phys_path = convert_physics_csv(args.physics_csv, phys_out)
    print(f"[OK] Physics LUT → {phys_path}")

if __name__ == "__main__":
    main()
