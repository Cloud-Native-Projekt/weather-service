"""Weather Service Maintenance Module

This module implements the daily and weekly maintenance routines for the weather service.
It handles data rollover, forecasting updates, and database maintenance operations.

Daily Maintenance Operations:
- Retrieves historical weather data for the rollover period (2 days ago)
- Truncates and refreshes daily weather forecast data
- Updates the database with new historical and forecast data

Weekly Maintenance Operations (Mondays only):
- Rolls over weekly forecast data to historical data for the previous week
- Generates new weekly forecast data from daily forecasts
- Maintains data integrity across weekly boundaries

The maintenance service uses the OpenMeteo API clients to fetch weather data
and the WeatherDatabase to manage data persistence. It automatically handles
rate limiting and error recovery during API operations.

Dependencies:
- OpenMeteo API clients for historical and forecast data retrieval
- WeatherDatabase for data persistence and management
- Pandas for data manipulation and aggregation

Usage:
    This module is typically run as a scheduled (daily) job to ensure
    the weather database remains current with fresh forecasts and properly
    archived historical data.

Example:
    python maintenance.py

Note:
    Weekly maintenance only runs on Mondays (weekday == 0) to process the
    previous week's data that has completed.
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
    WeeklyWeatherForecast,
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

        database.truncate_table(DailyWeatherForecast)

        ForecastClient = OpenMeteoForecastClient(config)
        forecast_data_daily = ForecastClient.main()
        forecast_orm_objects_daily = database.create_orm_objects(
            data=forecast_data_daily, table=DailyWeatherForecast
        )

        database.write_data(history_orm_objects_daily)
        database.write_data(forecast_orm_objects_daily)

        logger.info("Daily maintenance routine completed successfully!")

        if TODAY.weekday() == 0:
            logger.info("Starting weekly maintenance job...")

            rollover_year = (TODAY - timedelta(days=1)).year
            rollover_week = (TODAY - timedelta(days=1)).isocalendar().week

            logger.info(f"Rollover week is: {rollover_week}-{rollover_year}")

            database.rollover_weekly_data(rollover_year, rollover_week)

            forecast_data_weekly, _, _ = WeeklyTableConstructor().main(
                forecast_data_daily
            )
            forecast_data_weekly["source"] = "Open Meteo"
            forecast_orm_objects_weekly = database.create_orm_objects(
                data=forecast_data_weekly, table=WeeklyWeatherForecast
            )

            database.write_data(forecast_orm_objects_weekly)

            logger.info("Weekly maintenance routine completed successfully!")
        else:
            logger.info("Skipping weekly maintenance job...")

        logger.info("Maintenance routine completed successfully!")
    except Exception as e:
        logger.error(e)
    finally:
        database.close()
