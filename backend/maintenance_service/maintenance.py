import logging
from datetime import date, timedelta

from openmeteo_client import (
    OpenMeteoArchiveClient,
    OpenMeteoClientConfig,
    OpenMeteoForecastClient,
    WeeklyTableConstructor,
)

from models import (
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

        forecast_data_weekly, _, _ = WeeklyTableConstructor().main(forecast_data_daily)
        forecast_data_weekly["source"] = "Open Meteo"
        forecast_orm_objects_weekly = database.create_orm_objects(
            data=forecast_data_weekly, table=WeeklyWeatherForecast
        )

        database.write_data(forecast_orm_objects_weekly)

        logger.info("Weekly maintenance routine completed successfully!")
    else:
        logger.info("Skipping weekly maintenance job...")

    database.close()

    logger.info("Maintenance routine completed successfully!")
