-- Avertissement (pas d'échec) : lignes dont l'indicateur de panne contredit les types de panne.
{{ config(severity='warn') }}

select udi, machine_failure, twf, hdf, pwf, osf, rnf
from {{ ref('stg_sensor_readings') }}
where not is_label_consistent
