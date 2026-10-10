"""Étape 2 : explorer la couche raw dans DuckDB ."""
import duckdb

con = duckdb.connect("ai4i.duckdb", read_only=True)
print(con.sql("SELECT COUNT(*) AS lignes FROM raw.sensor_readings"))
con.close()

