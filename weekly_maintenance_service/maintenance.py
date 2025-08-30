"""Weather Service Weekly Maintenance Module

This module implements the weekly maintenance routines for the weather service.
It handles the aggregation of daily historical data into weekly summaries and
updates the WeeklyWeatherHistory table with newly completed weekly data.

Weekly Maintenance Operations:
- Executes only on Wednesdays (weekday == 2) to process the previous completed week
- Retrieves daily historical weather data for the previous week (Monday to Sunday)
- Aggregates daily data into weekly summaries using WeeklyTableConstructor
- Calculates weekly weather statistics from completed daily historical records
- Updates WeeklyWeatherHistory table with new weekly aggregated data
- Ensures weekly data is available for machine learning model training

The maintenance service processes the previous week's data to ensure completeness,
as weather data for the current week may still be incoming. The weekly aggregation
process converts daily weather observations into statistical summaries suitable
for weekly forecasting models.

Weekly Data Processing:
- Calculates date range for the previous completed week (Monday to Sunday)
- Fetches all daily historical data within the target week
- Applies WeeklyTableConstructor aggregation algorithms
- Generates weekly statistical summaries (means, maxima, minima, totals)
- Persists aggregated weekly data to WeeklyWeatherHistory table

Scheduling Logic:
- Runs only on Wednesdays to allow sufficient time for daily data completion
- Processes the week ending on the previous Sunday
- Skips execution on non-Wednesday days
- Ensures weekly data is available for subsequent model training operations

Dependencies:
- WeatherDatabase for data retrieval, conversion, and persistence
- WeeklyTableConstructor for daily-to-weekly data aggregation algorithms
- DailyWeatherHistory and WeeklyWeatherHistory ORM models for data access

Usage:
    This module is designed to run as a scheduled job, but only executes
    meaningful operations on Wednesdays. Can be safely run daily as it includes
    built-in day-of-week checking to prevent unnecessary processing.

Example:
    python maintenance.py

Data Flow:
    DailyWeatherHistory (previous week) → WeeklyTableConstructor → WeeklyWeatherHistory

Output:
    Updated WeeklyWeatherHistory table with aggregated data for the previous completed
    week, ready for use by model building and forecasting services.

Note:
    The service processes data for the week ending on the previous Sunday to ensure
    all daily data for that week is complete and stable. Weekly data is essential
    for training location-specific forecasting models.
"""

import logging
from datetime import date, timedelta

from openmeteo_client import WeeklyTableConstructor
from weather_models import DailyWeatherHistory, WeatherDatabase, WeeklyWeatherHistory

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Weekly Maintenance Service")

    database = WeatherDatabase()

    try:
        logger.info("Starting weekly maintenance job...")

        TODAY = date.today()

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
