-- Couche staging : données nettoyées, avec un indicateur de cohérence des étiquettes de panne.
select
    udi,
    product_id,
    quality_level,
    air_temperature_k,
    process_temperature_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
    machine_failure,
    twf, hdf, pwf, osf, rnf,
    not (
        (machine_failure = 1 and twf + hdf + pwf + osf + rnf = 0)
        or (machine_failure = 0 and twf + hdf + pwf + osf + rnf > 0)
    ) as is_label_consistent
from {{ source('raw', 'sensor_readings') }}
where udi is not null
