-- Échec si une mesure a une valeur impossible (température, vitesse, couple ou usure).
select udi
from {{ ref('stg_sensor_readings') }}
where air_temperature_k <= 0
   or process_temperature_k <= 0
   or rotational_speed_rpm <= 0
   or torque_nm < 0
   or tool_wear_min < 0
