"""Weather Service Forecast Generation Module

This module implements the weekly weather forecast generation routines for the weather service.
It handles the loading of trained machine learning models and generation of future weather
forecasts using historical data patterns stored in the database.

Forecast Generation Operations:
- Retrieves all unique locations from the WeeklyWeatherHistory table
- Loads pre-trained WeeklyForecastModel instances for each location
- Fetches historical weekly weather data for model input
- Generates weather forecasts using trained models with configurable horizon
- Truncates and refreshes WeeklyWeatherForecast table with new predictions
- Persists forecast data to the database for consumption by client applications

The forecast service processes each location independently, using location-specific
trained models to generate accurate weekly weather predictions. The forecast horizon
is configurable through the config.json file.

Forecast Generation Features:
- Location-specific model loading for tailored predictions
- Configurable forecast horizon through external configuration
- Automatic data retrieval and preprocessing from WeeklyWeatherHistory
- Robust error handling for individual model failures without stopping the entire process
- Complete refresh of forecast data to ensure consistency

Dependencies:
- WeatherDatabase for historical data retrieval and forecast persistence
- WeeklyForecastModel for loading trained models and generating predictions
- WeeklyWeatherHistory and WeeklyWeatherForecast ORM models for data access
- Configuration file (config.json) for forecast parameters

Usage:
    This module is designed to run periodically to generate fresh weather forecasts
    using the latest available historical data and trained models. Should run after
    model building and maintenance routines have completed.

Example:
    python forecast.py

Scheduling:
    Recommended to run weekly after maintenance routines to provide up-to-date forecasts.
    Should run after sufficient historical data updates and model availability.

Configuration:
    Forecast horizon is controlled via config.json file in the working directory.
    Models are loaded from the ./models directory in the current working directory.

Output:
    Generated forecasts are stored in the WeeklyWeatherForecast table and available
    for consumption by weather service clients and applications.

Note:
    The forecast generation process depends on the availability of trained models
    for each location. Individual forecast failures are logged but do not stop
    the generation of forecasts for other locations.
"""

import json
import logging
import os

from weather_models import (
    WeatherDatabase,
    WeeklyForecastModel,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
)

LOGLEVEL = os.getenv("LOGLEVEL", "INFO")

if __name__ == "__main__":
    logging.basicConfig(
        level=LOGLEVEL,
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
