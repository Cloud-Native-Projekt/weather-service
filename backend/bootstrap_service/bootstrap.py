import logging

import pandas as pd
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
    WeeklyWeatherHistory,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Init Service")

    database = WeatherDatabase()

    if database.bootstrap:
        logging.info("Starting bootstrap routine...")

        database.create_tables()
        database.truncate_table(DailyWeatherHistory)
        database.truncate_table(DailyWeatherForecast)
        database.truncate_table(WeeklyWeatherHistory)
        database.truncate_table(WeeklyWeatherForecast)

        config = OpenMeteoClientConfig(create_from_file=True)

        ArchiveClient = OpenMeteoArchiveClient(config)
        historic_data_daily = ArchiveClient.main()
        history_orm_objects_daily = database.create_orm_objects(
            data=historic_data_daily, table=DailyWeatherHistory
        )

        ForecastClient = OpenMeteoForecastClient(config)
        forecast_data_daily = ForecastClient.main()
        forecast_orm_objects_daily = database.create_orm_objects(
            data=forecast_data_daily, table=DailyWeatherForecast
        )

        historic_data_weekly, _, historic_data_daily_tail = (
            WeeklyTableConstructor().main(historic_data_daily)
        )
        history_orm_objects_weekly = database.create_orm_objects(
            data=historic_data_weekly, table=WeeklyWeatherHistory
        )

        forecast_data_weekly, _, _ = WeeklyTableConstructor().main(
            pd.concat(
                [historic_data_daily_tail, forecast_data_daily], axis=0, join="outer"
            )
        )
        forecast_data_weekly["source"] = "Open Meteo"
        forecast_orm_objects_weekly = database.create_orm_objects(
            data=forecast_data_weekly, table=WeeklyWeatherForecast
        )

        database.write_data(history_orm_objects_daily)
        database.write_data(forecast_orm_objects_daily)
        database.write_data(history_orm_objects_weekly)
        database.write_data(forecast_orm_objects_weekly)

        logging.info("Bootstrap routine completed successfully!")

    else:
        logging.info("Tables already exist. Skipping bootstrap routine...")

    database.close()
