import duckdb

con = duckdb.connect("ai4i.duckdb", read_only=True)

print("Colonnes de raw.sensor_readings :")
print(con.sql("DESCRIBE raw.sensor_readings").df()[["column_name"]].T.to_string())

print("\nTaux de panne et types de panne par qualité :")
print(con.sql("""
    SELECT quality_level, COUNT(*) AS n, SUM(machine_failure) AS pannes,
           ROUND(100*AVG(machine_failure), 2) AS taux_pct,
           SUM(twf) AS twf, SUM(hdf) AS hdf, SUM(pwf) AS pwf, SUM(osf) AS osf
    FROM raw.sensor_readings GROUP BY 1 ORDER BY 1""").df())

print("\nPlus petit usure x couple parmi les pannes de surcharge :")
print(con.sql("""
    SELECT quality_level, ROUND(MIN(tool_wear_min * torque_nm)) AS min_wear_x_torque
    FROM raw.sensor_readings WHERE osf = 1 GROUP BY 1 ORDER BY 1""").df())