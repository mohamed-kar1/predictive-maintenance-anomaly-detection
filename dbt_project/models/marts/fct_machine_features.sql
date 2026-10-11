-- Couche marts : table de variables prête pour l'entraînement.
-- 6 mesures + 3 variables calculées (temp_diff_k, power_w, wear_x_torque).
select
    udi,
    machine_failure,
    case quality_level when 'L' then 0 when 'M' then 1 when 'H' then 2 end as quality_level_ord,
    air_temperature_k,
    process_temperature_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
    process_temperature_k - air_temperature_k as temp_diff_k,
    torque_nm * rotational_speed_rpm * 2 * pi() / 60 as power_w,
    tool_wear_min * torque_nm as wear_x_torque,
    is_label_consistent
from {{ ref('stg_sensor_readings') }}
