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
        logger.error(f"{e}")
    finally:
        database.close()
