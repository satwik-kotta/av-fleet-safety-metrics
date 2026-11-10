# Databricks notebook source
# MAGIC %md
# MAGIC # NGSIM US-101 ingest on Databricks
# MAGIC
# MAGIC The same ingest as `src/fdb/ingest.py`, written as a Databricks notebook. It reads the three NGSIM US-101
# MAGIC periods, converts feet to metres, runs the data checks, joins every car to its leader to get gap, closing speed
# MAGIC and time to collision, and saves a Delta table. The last cells count hard brakes and near misses per 1,000
# MAGIC vehicle km in Spark SQL.
# MAGIC
# MAGIC **To run on Databricks Free Edition:** run `scripts/get_data.sh` locally, create a Volume (Catalog, then
# MAGIC Create, then Volume) and upload the three `*_us101_smoothed_11_.csv` files into it. Set `VOLUME` below to the
# MAGIC Volume path and run all cells.

# COMMAND ----------

VOLUME = "/Volumes/workspace/default/ngsim"   # where the three CSV files were uploaded
TABLE = "workspace.default.ngsim_us101_frames"
FT = 0.3048
PERIODS = ["0750_0805", "0805_0820", "0820_0835"]

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

dfs = [spark.read.csv(f"{VOLUME}/{p}_us101_smoothed_11_.csv", header=True, inferSchema=True).withColumn("period", F.lit(p))
       for p in PERIODS]
raw = dfs[0].unionByName(dfs[1]).unionByName(dfs[2])
df = raw.select(
    "period", F.col("Vehicle_ID").alias("vid"), F.col("Frame_ID").alias("frame"), F.col("Total_Frames").alias("total_frames"),
    (F.col("Local_X") * FT).alias("x"), (F.col("Local_Y") * FT).alias("y"), (F.col("v_Length") * FT).alias("length"),
    F.col("v_Class").alias("vclass"), (F.col("v_Vel") * FT).alias("v"), (F.col("v_Acc") * FT).alias("a"),
    F.col("Lane_ID").alias("lane"), F.col("Preceeding").alias("leader"), (F.col("Space_Hdwy") * FT).alias("space_headway"),
    F.col("Time_Hdwy").alias("time_headway"))
display(df.groupBy("period").count())   # each file stops at 1,048,575 rows, the Excel row limit

# COMMAND ----------

# MAGIC %md ## Data checks

# COMMAND ----------

per = df.groupBy("period", "vid").agg(F.count("*").alias("n"), F.min("frame").alias("f0"), F.max("frame").alias("f1"),
                                      F.first("total_frames").alias("tot"))
checks = {
    "rows": df.count(),
    "duplicate_vehicle_frames": df.count() - df.select("period", "vid", "frame").distinct().count(),
    "vehicles_with_frame_gaps": per.filter(F.col("n") != F.col("f1") - F.col("f0") + 1).count(),
    "vehicles_cut_short_by_row_limit": per.filter(F.col("n") != F.col("tot")).count(),
    "accel_over_8_mps2": df.filter(F.abs("a") > 8).count(),
}
ids = df.select("period", F.col("vid").alias("leader")).distinct().withColumn("present", F.lit(1))
checks["rows_with_missing_leader"] = (df.filter(F.col("leader") > 0).join(ids, ["period", "leader"], "left")
                                      .filter(F.col("present").isNull()).count())
print(checks)

# COMMAND ----------

# MAGIC %md ## Leader join, time to collision and lane history

# COMMAND ----------

df = df.join(per.filter(F.col("n") == F.col("tot")).select("period", "vid"), ["period", "vid"])
lead = df.select("period", "frame", F.col("vid").alias("leader"), F.col("v").alias("v_lead"),
                 F.col("length").alias("len_lead"), F.col("lane").alias("lane_lead"))
w = Window.partitionBy("period", "vid").orderBy("frame")
frames = (df.join(lead, ["period", "frame", "leader"], "left")
          .withColumn("gap", F.col("space_headway") - F.col("len_lead"))
          .withColumn("closing", F.col("v") - F.col("v_lead"))
          .withColumn("leader_missing", (F.col("leader") > 0) & F.col("v_lead").isNull())
          .withColumn("ttc", F.when((F.col("closing") > 0.1) & (F.col("gap") > 0), F.col("gap") / F.col("closing")))
          .withColumn("prev_lane", F.lag("lane").over(w))
          .withColumn("lat_v", (F.col("x") - F.lag("x", 5).over(w)) / 0.5))
frames.write.mode("overwrite").option("overwriteSchema", "true").partitionBy("period").saveAsTable(TABLE)

# COMMAND ----------

# MAGIC %md ## Event rates per 1,000 vehicle km (mainline lanes 1 to 5)

# COMMAND ----------

display(spark.sql(f"""
WITH f AS (
  SELECT *,
         a < -3 AS braking,
         NOT leader_missing AND lane = lane_lead AND ttc < 1.5 AS close_call,
         LAG(a < -3) OVER (PARTITION BY period, vid ORDER BY frame) AS braking_before,
         LAG(NOT leader_missing AND lane = lane_lead AND ttc < 1.5) OVER (PARTITION BY period, vid ORDER BY frame) AS close_before
  FROM {TABLE} WHERE lane BETWEEN 1 AND 5
)
SELECT period,
       ROUND(SUM(v) * 0.1 / 1000, 1)                                              AS vehicle_km,
       ROUND(AVG(v), 2)                                                           AS mean_speed_mps,
       ROUND(1000 * SUM(CASE WHEN braking AND NOT COALESCE(braking_before, false) THEN 1 ELSE 0 END)
             / (SUM(v) * 0.1 / 1000), 1)                                          AS braking_onsets_per_1000_km,
       ROUND(1000 * SUM(CASE WHEN close_call AND NOT COALESCE(close_before, false) THEN 1 ELSE 0 END)
             / (SUM(v) * 0.1 / 1000), 1)                                          AS ttc_under_1_5s_onsets_per_1000_km
FROM f GROUP BY period ORDER BY period
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC These SQL counts use every onset, without the minimum durations and the plausibility filter in
# MAGIC `src/fdb/events.py`, so they run higher than the rates in the dashboard. They show the same trend: near misses
# MAGIC rise sharply as the freeway slows down.
