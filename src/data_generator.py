"""
Simulate realistic TfL cycle hire + Met Office weather data.
Produces ~3 years of hourly station-level demand (25M+ trips compressed to hourly counts).
"""

import numpy as np
import pandas as pd
import holidays


def generate_tfl_bike_data(
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    n_stations: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h")
    uk_holidays = holidays.UK(years=range(2021, 2024))

    station_ids = [f"S{i:03d}" for i in range(1, n_stations + 1)]
    # Station capacity varies: central London stations busier
    station_capacity = rng.integers(15, 50, size=n_stations)
    station_type = rng.choice(["commuter", "leisure", "mixed"], size=n_stations, p=[0.4, 0.3, 0.3])

    records = []
    for sid, cap, stype in zip(station_ids, station_capacity, station_type):
        demand = _station_demand(hours, cap, stype, uk_holidays, rng)
        df = pd.DataFrame({"timestamp": hours, "station_id": sid, "demand": demand})
        records.append(df)

    return pd.concat(records, ignore_index=True)


def _station_demand(
    hours: pd.DatetimeIndex,
    capacity: int,
    station_type: str,
    uk_holidays,
    rng: np.random.Generator,
) -> np.ndarray:
    h = hours.hour.values
    dow = hours.dayofweek.values  # 0=Mon
    month = hours.month.values
    is_weekend = (dow >= 5).astype(float)
    is_holiday = np.array([d in uk_holidays for d in hours.date], dtype=float)

    # Hour-of-day profile
    if station_type == "commuter":
        am_peak = np.exp(-0.5 * ((h - 8) / 1.2) ** 2)
        pm_peak = np.exp(-0.5 * ((h - 17) / 1.2) ** 2)
        hourly_base = 0.3 * am_peak + 0.4 * pm_peak + 0.05
        # Weekends: flatten commute peaks
        weekend_mask = (is_weekend + is_holiday).clip(0, 1).astype(bool)
        leisure_profile = np.exp(-0.5 * ((h - 13) / 3.0) ** 2) * 0.4 + 0.05
        hourly_base[weekend_mask] = leisure_profile[weekend_mask]
    elif station_type == "leisure":
        hourly_base = np.exp(-0.5 * ((h - 13) / 3.5) ** 2) * 0.5 + 0.03
    else:  # mixed
        am_peak = np.exp(-0.5 * ((h - 8) / 1.5) ** 2) * 0.25
        pm_peak = np.exp(-0.5 * ((h - 17) / 1.5) ** 2) * 0.3
        midday = np.exp(-0.5 * ((h - 13) / 2.5) ** 2) * 0.25
        hourly_base = am_peak + pm_peak + midday + 0.04

    # Seasonality (summer peak)
    seasonal = 0.6 + 0.4 * np.sin((month - 3) * np.pi / 6).clip(0, 1)

    base_demand = hourly_base * seasonal * capacity * 0.6
    noise = rng.normal(0, base_demand * 0.15 + 0.3)
    demand = (base_demand + noise).clip(0, capacity)
    return np.round(demand).astype(int)


def generate_weather_data(
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    hours = pd.date_range(start, end, freq="h")
    month = hours.month.values
    h = hours.hour.values

    # Temperature: seasonal + diurnal cycle + noise
    temp_seasonal = 10 + 9 * np.sin((month - 1) * np.pi / 6 - np.pi / 2)
    temp_diurnal = 3 * np.sin((h - 6) * np.pi / 12)
    temperature = temp_seasonal + temp_diurnal + rng.normal(0, 2, len(hours))

    # Rainfall: occasional events, heavier in autumn/winter
    rain_prob = 0.12 + 0.06 * np.cos((month - 7) * np.pi / 6)
    rain_event = rng.random(len(hours)) < rain_prob
    rainfall = np.where(rain_event, rng.exponential(2.5, len(hours)), 0.0).round(1)

    # Wind speed
    wind_seasonal = 5 + 2 * np.cos((month - 1) * np.pi / 6)
    wind_speed = (wind_seasonal + rng.weibull(2, len(hours)) * 3).round(1)

    # Daylight hours (London latitude ~51.5°N)
    day_of_year = hours.dayofyear.values
    daylight = 7.5 + 4.5 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    sunrise = 12 - daylight / 2
    sunset = 12 + daylight / 2
    is_daylight = ((h >= np.floor(sunrise)) & (h <= np.ceil(sunset))).astype(int)

    return pd.DataFrame({
        "timestamp": hours,
        "temperature_c": temperature.round(1),
        "rainfall_mm": rainfall,
        "wind_speed_kmh": wind_speed,
        "daylight_hours": daylight.round(2),
        "is_daylight": is_daylight,
    })
