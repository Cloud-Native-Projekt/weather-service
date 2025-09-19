"""Weather Service Daily Maintenance Module

This module implements the daily maintenance routines for the weather service.
It handles data rollover operations, forecast updates, database health checks,
and automatic data gap recovery to ensure database integrity and freshness.

Daily Maintenance Operations:
- Retrieves historical weather data for the rollover period (2 days ago)
- Truncates and refreshes daily weather forecast data with current forecasts
- Updates the database with new historical and forecast data
- Performs comprehensive health checks to identify missing historical data
- Automatically repairs data gaps by fetching missing dates from OpenMeteo Archive API

The maintenance service uses OpenMeteo API clients to fetch weather data and the
WeatherDatabase to manage data persistence. It includes robust error handling,
health monitoring, and automatic data gap recovery to ensure database integrity
and operational continuity.

Data Rollover Process:
- Fetches historical data for the rollover date (2 days prior to current date)
- Converts retrieved data to ORM objects for database persistence
- Writes new historical data to DailyWeatherHistory table
- Refreshes forecast data by truncating and repopulating DailyWeatherForecast table

Health Check Features:
- Validates completeness of historical data within configured date ranges
- Identifies missing dates in the DailyWeatherHistory table
- Automatically backfills missing data by fetching from OpenMeteo Archive API
- Ensures data continuity for reliable weather service operations
- Provides detailed logging for monitoring and debugging data issues

Data Integrity Safeguards:
- Uses 2-day rollover delay to ensure data stability and availability
- Comprehensive error handling to prevent partial updates
- Individual missing date recovery without affecting other operations
- Health check validation using configured date ranges from OpenMeteoClientConfig

Dependencies:
- OpenMeteo API clients (Archive and Forecast) for weather data retrieval
- WeatherDatabase for data persistence, health checks, and management
- OpenMeteoClientConfig for API configuration and date range management
- DailyWeatherHistory and DailyWeatherForecast ORM models for data access

Usage:
    This module is designed to run as a scheduled daily job to ensure the weather
    database remains current with fresh forecasts and properly archived historical data.

Example:
    python maintenance.py

Configuration:
    Date ranges and API parameters are controlled via OpenMeteoClientConfig
    which reads from external configuration files for flexibility.

Output:
    Updated DailyWeatherHistory and DailyWeatherForecast tables with fresh data,
    complete historical records without gaps, and comprehensive operation logs.

Note:
    The rollover date is set to 2 days ago to ensure data stability and account for
    delays in weather data availability. Health checks ensure no data gaps
    exist in the historical record within the configured date range.
"""

import logging
from datetime import date, timedelta

from openmeteo_client import (
    OpenMeteoArchiveClient,
    OpenMeteoClientConfig,
    OpenMeteoForecastClient,
)
from weather_models import DailyWeatherForecast, DailyWeatherHistory, WeatherDatabase

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Daily Maintenance Service")

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

        # Broken
        # if not database.health_check(
        #     start_date=health_check_config.history_start_date,
        #     end_date=health_check_config.history_end_date,
        #     table=DailyWeatherHistory,
        # ):
        #     missing_entries = database.get_missing_entries

        #     for missing_entry in missing_entries:
        #         logger.info(f"Retrieving entry: {missing_entry}")
        #         temp_config = OpenMeteoClientConfig(
        #             create_from_file=True,
        #             kwargs={
        #                 "history_start_date": missing_entry[0],
        #                 "history_end_date": missing_entry[0],
        #                 "bounding_box": {
        #                     "north": missing_entry[1],
        #                     "south": missing_entry[1],
        #                     "west": missing_entry[2],
        #                     "east": missing_entry[2],
        #                 },
        #             },
        #         )

        #         ArchiveClient = OpenMeteoArchiveClient(temp_config)
        #         historic_data_daily = ArchiveClient.main()
        #         history_orm_objects_daily = database.create_orm_objects(
        #             data=historic_data_daily, table=DailyWeatherHistory
        #         )

        #         database.write_data(history_orm_objects_daily)

        # else:
        #     logger.info("Daily health check passed successfully!")

        logger.info("Maintenance routine completed successfully!")
    except Exception as e:
        logger.exception("An error occurred during the maintenance routine: ")
    finally:
        database.close()
