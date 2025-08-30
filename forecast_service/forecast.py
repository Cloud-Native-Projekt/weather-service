import json
import logging
import os

from weather_models import (
    WeatherDatabase,
    WeeklyForecastModel,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Forecast Service")

    try:
        logger.info("Starting forecast service...")

        database = WeatherDatabase()

        database.truncate_table(WeeklyWeatherForecast)

        locations = database.get_locations(WeeklyWeatherHistory)

        logger.info(f"Running {len(locations)} models for {locations}")

        for location in locations:
            try:
                with open(
                    file=(os.path.join(os.getcwd(), "config.json")), mode="r"
                ) as file:
                    config = json.load(fp=file)
                    file.close()

                model = WeeklyForecastModel.from_file(
                    directory=(os.path.join(os.getcwd(), "models")), location=location
                )

                weekly_weather_history = database.get_data_by_location(
                    table=WeeklyWeatherHistory, location=location
                )

                weekly_weather_history = database.to_dataframe(weekly_weather_history)

                forecast = model.forecast(
                    horizon=config["horizon"], data=weekly_weather_history
                )

                forecast_orm_objects = database.create_orm_objects(
                    data=forecast, table=WeeklyWeatherForecast
                )

                database.write_data(forecast_orm_objects)

                logger.info("Weekly forecasts created successfully!")
            except Exception as e:
                logger.exception(
                    f"An error occurred while building the model for {location}:"
                )

    except Exception as e:
        logger.exception("An error occurred in the forecast service:")
