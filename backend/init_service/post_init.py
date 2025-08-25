import logging

import pandas as pd

from models import (
    DailyWeatherForecast,
    DailyWeatherHistory,
    WeatherDatabase,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Post Init Service")

    logger.info("Beginning Post Init Script...")

    database = WeatherDatabase()

    # Get daily data
    logger.info("Retrieving daily data from database...")
    daily_history = pd.DataFrame(
        [row.__dict__ for row in database.get_table(DailyWeatherHistory)]
    )
    daily_forecast = pd.DataFrame(
        [row.__dict__ for row in database.get_table(DailyWeatherForecast)]
    )

    # Convert date column to pandas datetime
    daily_history["date"] = pd.to_datetime(daily_history["date"])
    daily_forecast["date"] = pd.to_datetime(daily_forecast["date"])

    # Calculate cutoff dates (First Monday to last Sunday)
    weekly_history_cutoff_front = daily_history[daily_history["date"].dt.weekday == 0][
        "date"
    ].min()
    weekly_history_cutoff_back = daily_history[daily_history["date"].dt.weekday == 6][
        "date"
    ].max()
    weekly_forecast_cutoff_back = daily_forecast[
        daily_forecast["date"].dt.weekday == 6
    ]["date"].max()

    logger.info(
        f"Full weekly history available from {weekly_history_cutoff_front} to {weekly_history_cutoff_back}\nFull weekly forecast available until {weekly_forecast_cutoff_back}"
    )

    logger.info("Building tables...")

    # Cutoff data outside cutoff dates
    weekly_history = daily_history[
        (daily_history["date"] >= weekly_history_cutoff_front)
        & (daily_history["date"] <= weekly_history_cutoff_back)
    ].copy()
    weekly_history_tail = daily_history[
        daily_history["date"] > weekly_history_cutoff_back
    ].copy()
    weekly_forecast = daily_forecast[
        daily_forecast["date"] <= weekly_forecast_cutoff_back
    ].copy()

    # Append weekly history tail to weekly forecast data
    weekly_forecast = pd.concat(
        [weekly_history_tail, weekly_forecast], axis=0, join="outer"
    )

    # Create columns for week and year
    weekly_history["week"] = weekly_history["date"] + pd.offsets.Week(weekday=6)
    weekly_history["year"] = weekly_history["week"].dt.year
    weekly_history = weekly_history.drop(columns="date")
    weekly_forecast["week"] = weekly_forecast["date"] + pd.offsets.Week(weekday=6)
    weekly_forecast["year"] = weekly_forecast["week"].dt.year
    weekly_forecast = weekly_forecast.drop(columns="date")

    # Group data by week and location
    weekly_history = (
        weekly_history.groupby(["week", "year", "latitude", "longitude"])
        .aggregate(
            {
                "temperature_2m_mean": "mean",
                "temperature_2m_max": "max",
                "temperature_2m_min": "min",
                "cloud_cover_mean": "mean",
                "cloud_cover_max": "max",
                "cloud_cover_min": "min",
                "wind_speed_10m_mean": "mean",
                "wind_speed_10m_min": "min",
                "wind_speed_10m_max": "max",
                "sunshine_duration": "mean",
                "precipitation_sum": "sum",
                "precipitation_hours": "sum",
            }
        )
        .reset_index()
    )

    logger.info(f"Created Weekly History Table with {len(weekly_history)} entries")

    weekly_forecast = (
        weekly_forecast.groupby(["week", "year", "latitude", "longitude"])
        .aggregate(
            {
                "temperature_2m_mean": "mean",
                "temperature_2m_max": "max",
                "temperature_2m_min": "min",
                "cloud_cover_mean": "mean",
                "cloud_cover_max": "max",
                "cloud_cover_min": "min",
                "wind_speed_10m_mean": "mean",
                "wind_speed_10m_min": "min",
                "wind_speed_10m_max": "max",
                "sunshine_duration": "mean",
                "precipitation_sum": "sum",
                "precipitation_hours": "sum",
            }
        )
        .reset_index()
    )

    # Retrieve calendar week
    weekly_history["week"] = weekly_history["week"].dt.isocalendar().week
    weekly_forecast["week"] = weekly_forecast["week"].dt.isocalendar().week

    # Add source column to forecast table
    weekly_forecast["source"] = "Open Meteo"

    logger.info(f"Created Weekly Forecast Table with {len(weekly_forecast)} entries")

    # Write data to db
    logger.info("Writing data to database...")
    weekly_history_orm_objects = database.create_orm_objects(
        data=weekly_history, table=WeeklyWeatherHistory
    )
    weekly_forecast_orm_objects = database.create_orm_objects(
        data=weekly_forecast, table=WeeklyWeatherForecast
    )

    database.write_data(weekly_history_orm_objects)
    database.write_data(weekly_forecast_orm_objects)

    database.close()

    logger.info("Post init script excited successfully!")
