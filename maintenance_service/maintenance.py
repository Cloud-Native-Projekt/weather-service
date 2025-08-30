"""Weather Service Maintenance Module

This module implements the daily and weekly maintenance routines for the weather service.
It handles data rollover, forecasting updates, database health checks, and maintenance operations.

Daily Maintenance Operations:
- Retrieves historical weather data for the rollover period (2 days ago)
- Truncates and refreshes daily weather forecast data with current forecasts
- Updates the database with new historical and forecast data
- Performs health checks to identify and backfill missing historical data
- Automatically repairs data gaps by fetching missing dates from OpenMeteo Archive API

Weekly Maintenance Operations (Wednesdays only):
- Processes the previous week's completed daily historical data
- Generates weekly aggregated data from daily historical records
- Calculates weekly summaries using WeeklyTableConstructor
- Updates WeeklyWeatherHistory table with new weekly data

The maintenance service uses OpenMeteo API clients to fetch weather data and the
WeatherDatabase to manage data persistence. It includes robust error handling,
health monitoring, and automatic data gap recovery to ensure database integrity.

Health Check Features:
- Validates completeness of historical data within configured date ranges
- Identifies missing dates in the DailyWeatherHistory table
- Automatically backfills missing data by fetching from OpenMeteo Archive API
- Ensures data continuity for reliable weather service operations

Dependencies:
- OpenMeteo API clients (Archive and Forecast) for weather data retrieval
- WeatherDatabase for data persistence, health checks, and management
- WeeklyTableConstructor for daily-to-weekly data aggregation
- Weather model classes for ORM object creation

Usage:
    This module is designed to run as a scheduled daily job to ensure the weather
    database remains current with fresh forecasts and properly archived historical data.
    Weekly operations automatically trigger on Wednesdays.

Example:
    python maintenance.py

Scheduling:
    Daily: Recommended to run early morning (e.g., 2:00 AM) to ensure fresh data
    Weekly: Automatically runs on Wednesdays (weekday == 2) to process completed weeks

Note:
    The rollover date is set to 2 days ago to ensure data stability and account for
    delays in weather data availability. Health checks ensure no data gaps
    exist in the historical record.
"""

import logging
from datetime import date, timedelta

from openmeteo_client import (
    OpenMeteoArchiveClient,
    OpenMeteoClientConfig,
    OpenMeteoForecastClient,
    WeeklyTableConstructor,
)
from weather_models import (
    DailyWeatherForecast,
    DailyWeatherHistory,
    WeatherDatabase,
    WeeklyWeatherHistory,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Maintenance Service")

    database = WeatherDatabase()

    try:
        logger.info("Starting daily maintenance job...")

        TODAY = date.today()

        rollover_date = TODAY - timedelta(days=2)

        logger.info(f"Rollover date is: {rollover_date}")

        config = OpenMeteoClientConfig(
            create_from_file=True,
            kwargs={"history_start_date": rollover_date},
        )

        ArchiveClient = OpenMeteoArchiveClient(config)
        historic_data_daily = ArchiveClient.main()
        history_orm_objects_daily = database.create_orm_objects(
            data=historic_data_daily, table=DailyWeatherHistory
        )

        database.write_data(history_orm_objects_daily)

        database.truncate_table(DailyWeatherForecast)

        ForecastClient = OpenMeteoForecastClient(config)
        forecast_data_daily = ForecastClient.main()
        forecast_orm_objects_daily = database.create_orm_objects(
            data=forecast_data_daily, table=DailyWeatherForecast
        )

        database.write_data(forecast_orm_objects_daily)

        logger.info("Daily maintenance routine completed successfully!")

        logger.info("Performing daily health check...")

        health_check_config = OpenMeteoClientConfig(create_from_file=True)

        if not database.health_check(
            start_date=health_check_config.history_start_date,
            end_date=health_check_config.history_end_date,
            table=DailyWeatherHistory,
        ):
            missing_dates = database.get_missing_dates(
                start_date=health_check_config.history_start_date,
                end_date=health_check_config.history_end_date,
                table=DailyWeatherHistory,
            )

            for datum in missing_dates:
                temp_config = OpenMeteoClientConfig(
                    create_from_file=True,
                    kwargs={"history_start_date": datum, "history_end_date": datum},
                )

                ArchiveClient = OpenMeteoArchiveClient(temp_config)
                historic_data_daily = ArchiveClient.main()
                history_orm_objects_daily = database.create_orm_objects(
                    data=historic_data_daily, table=DailyWeatherHistory
                )

                database.write_data(history_orm_objects_daily)

        else:
            logger.info("Daily health check passed successfully!")

        if TODAY.weekday() == 2:
            logger.info("Starting weekly maintenance job...")

            start_date = TODAY - timedelta(days=TODAY.weekday()) - timedelta(days=7)
            end_date = TODAY - timedelta(days=TODAY.weekday()) - timedelta(days=1)

            historic_data_daily_last_week = database.get_data_by_date_range(
                table=DailyWeatherHistory, start_date=start_date, end_date=end_date
            )

            historic_data_daily_last_week = database.to_dataframe(
                historic_data_daily_last_week
            )

            historic_data_weekly, _, _ = WeeklyTableConstructor().main(
                historic_data_daily_last_week
            )
            history_orm_objects_weekly = database.create_orm_objects(
                data=historic_data_weekly, table=WeeklyWeatherHistory
            )

            database.write_data(history_orm_objects_weekly)

            logger.info("Weekly maintenance routine completed successfully!")
        else:
            logger.info("Skipping weekly maintenance job...")

        logger.info("Maintenance routine completed successfully!")
    except Exception as e:
        logger.exception("An error occurred during the maintenance routine: ")
    finally:
        database.close()
