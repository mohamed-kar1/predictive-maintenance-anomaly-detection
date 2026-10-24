import duckdb

con = duckdb.connect("ai4i.duckdb", read_only=True)
print(con.sql("""
    SELECT quality_level, COUNT(*) AS n,
           ROUND(AVG(process_temperature_k - air_temperature_k), 2) AS ecart_moyen,
           ROUND(STDDEV(process_temperature_k - air_temperature_k), 2) AS ecart_type,
           ROUND(MIN(process_temperature_k - air_temperature_k), 1) AS mini,
           ROUND(MAX(process_temperature_k - air_temperature_k), 1) AS maxi,
           SUM(hdf) AS pannes_chaleur,
           ROUND(MAX(CASE WHEN hdf = 1 THEN process_temperature_k - air_temperature_k END), 2) AS ecart_max_en_panne
    FROM raw.sensor_readings GROUP BY 1 ORDER BY 1""").df().to_string(index=False))