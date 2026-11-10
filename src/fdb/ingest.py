"""Ingest NGSIM US-101 trajectories with PySpark.

Steps (all in Spark):
  1. read the three 15 minute periods, convert feet to metres
  2. data checks (duplicates, frame gaps, impossible values)
  3. measure the truncation: each published file stops at 1,048,575 rows, the Excel
     row limit, so the vehicles with the highest IDs are missing. The one vehicle cut
     mid trajectory is dropped, and rows whose leader is a missing vehicle get no gap or TTC.
  4. join each vehicle to its leader in the same frame to get the gap, closing speed
     and time to collision (TTC)
  5. window functions for previous lane and smoothed lateral speed
  6. write one Parquet table partitioned by period
"""
from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "ngsim" / "raw"
OUT = ROOT / "data" / "ngsim" / "frames.parquet"
FT = 0.3048
PERIODS = {"0750_0805": "7:50 to 8:05", "0805_0820": "8:05 to 8:20", "0820_0835": "8:20 to 8:35"}
EXCEL_ROW_LIMIT = 1_048_575
MAINLINE = [1, 2, 3, 4, 5]


def spark() -> SparkSession:
    return (SparkSession.builder.master("local[*]").appName("ngsim")
            .config("spark.sql.shuffle.partitions", "16").config("spark.driver.memory", "6g")
            .config("spark.sql.session.timeZone", "UTC").getOrCreate())


def read(sp: SparkSession):
    dfs = []
    for key in PERIODS:
        f = str(RAW / f"{key}_us101_smoothed_11_.csv")
        dfs.append(sp.read.csv(f, header=True, inferSchema=True).withColumn("period", F.lit(key)))
    df = dfs[0].unionByName(dfs[1]).unionByName(dfs[2])
    return df.select(
        "period", F.col("Vehicle_ID").alias("vid"), F.col("Frame_ID").alias("frame"),
        F.col("Total_Frames").alias("total_frames"),
        (F.col("Local_X") * FT).alias("x"), (F.col("Local_Y") * FT).alias("y"),
        (F.col("v_Length") * FT).alias("length"), F.col("v_Class").alias("vclass"),
        (F.col("v_Vel") * FT).alias("v"), (F.col("v_Acc") * FT).alias("a"),
        F.col("Lane_ID").alias("lane"), F.col("Preceeding").alias("leader"),
        (F.col("Space_Hdwy") * FT).alias("space_headway"), F.col("Time_Hdwy").alias("time_headway"))


def truncation(df) -> dict:
    """Each file stops at the Excel row limit, so the vehicles with the highest IDs are
    missing. Two effects are measured: vehicles whose own trajectory is cut short (dropped),
    and rows whose leader is one of the missing vehicles (gap and TTC set to unknown)."""
    ids = df.select("period", F.col("vid").alias("leader")).distinct().withColumn("present", F.lit(1))
    j = df.filter(F.col("leader") > 0).join(ids, ["period", "leader"], "left")
    miss = j.filter(F.col("present").isNull())
    out = {}
    for r in df.groupBy("period").agg(F.count("*").alias("rows"), F.max("vid").alias("max_vid")).collect():
        out[r.period] = {"rows": r.rows, "max_vehicle_id": r.max_vid}
    for r in miss.groupBy("period").agg(F.countDistinct("leader").alias("ids"), F.count("*").alias("rows"),
                                        F.min("leader").alias("lo")).collect():
        out[r.period].update({"missing_leader_ids": r.ids, "rows_with_missing_leader": r.rows,
                              "lowest_missing_id": r.lo})
    return out


def checks(df) -> dict:
    agg = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("period", "vid").alias("vehicles"),
        F.sum((F.col("v") < 0).cast("int")).alias("negative_speed"),
        F.sum((F.col("v") > 45).cast("int")).alias("speed_over_45_mps"),
        F.sum((F.abs("a") > 8).cast("int")).alias("accel_over_8_mps2"),
    ).first().asDict()
    agg["duplicate_vehicle_frames"] = agg["rows"] - df.select("period", "vid", "frame").distinct().count()
    per = df.groupBy("period", "vid").agg(F.count("*").alias("n"), F.min("frame").alias("f0"),
                                          F.max("frame").alias("f1"), F.first("total_frames").alias("tot"))
    agg["vehicles_with_frame_gaps"] = per.filter(F.col("n") != F.col("f1") - F.col("f0") + 1).count()
    agg["vehicles_cut_short"] = per.filter(F.col("n") != F.col("tot")).count()
    agg["rows_per_file_at_excel_limit"] = [r["count"] for r in df.groupBy("period").count().orderBy("period").collect()]
    return agg


def enrich(df):
    # drop the one vehicle per file whose trajectory the row limit cut short
    per = df.groupBy("period", "vid").agg(F.count("*").alias("n"), F.first("total_frames").alias("tot"))
    df = df.join(per.filter(F.col("n") == F.col("tot")).select("period", "vid"), ["period", "vid"])
    lead = df.select("period", "frame", F.col("vid").alias("leader"), F.col("v").alias("v_lead"),
                     F.col("length").alias("len_lead"), F.col("lane").alias("lane_lead"))
    df = df.join(lead, ["period", "frame", "leader"], "left")
    df = (df.withColumn("gap", F.col("space_headway") - F.col("len_lead"))
            .withColumn("closing", F.col("v") - F.col("v_lead"))
            .withColumn("leader_missing", (F.col("leader") > 0) & F.col("v_lead").isNull())
            .withColumn("ttc", F.when((F.col("closing") > 0.1) & (F.col("gap") > 0), F.col("gap") / F.col("closing"))))
    w = Window.partitionBy("period", "vid").orderBy("frame")
    df = (df.withColumn("prev_lane", F.lag("lane").over(w))
            .withColumn("x_lag5", F.lag("x", 5).over(w))
            .withColumn("lat_v", (F.col("x") - F.col("x_lag5")) / 0.5))   # lateral speed over the last 0.5 s
    return df.drop("x_lag5")


def main():
    sp = spark()
    df = read(sp).cache()
    chk = checks(df)
    trunc = truncation(df)
    enrich(df).repartition("period").write.mode("overwrite").partitionBy("period").parquet(str(OUT))
    meta = {"checks": chk, "truncation": trunc, "excel_row_limit": EXCEL_ROW_LIMIT}
    (ROOT / "outputs" / "freeway").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "freeway" / "ingest.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))
    sp.stop()


if __name__ == "__main__":
    main()
