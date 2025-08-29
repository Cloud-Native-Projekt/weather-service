import logging
import os

from weather_models import WeatherDatabase, WeeklyForecastModel, WeeklyWeatherHistory

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Maintenance Service")

    try:
        logger.info("Starting model build service...")

        database = WeatherDatabase()

        locations = database.get_locations(WeeklyWeatherHistory)

        logger.info(f"Building {len(locations)} models for {locations}")

        for location in locations:
            try:
                model = WeeklyForecastModel(location)

                weekly_weather_history = database.get_data_by_location(
                    table=WeeklyWeatherHistory, location=location
                )

                weekly_weather_history = database.to_dataframe(weekly_weather_history)

                model.build_model(weekly_weather_history)

                model.save(os.path.join(os.getcwd(), "models"))
            except Exception as e:
                logger.error(f"{e}")

        logger.info("Weekly forecast models build successfully!")
    except Exception as e:
        logger.error(f"{e}")
