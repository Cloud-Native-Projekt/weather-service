"""Weather Service Model Building Module

This module implements the model building routines for the weather service.
It handles the creation and training of weekly weather forecast models using historical
weather data stored in the database.

Model Building Operations:
- Retrieves all unique locations from the WeeklyWeatherHistory table
- Iterates through each location to build location-specific forecast models
- Fetches historical weekly weather data for each location from the database
- Converts database records to dataframes for model training
- Trains WeeklyForecastModel instances using historical data patterns
- Saves trained models to the local models directory for use by the forecast service

The model building service processes each location independently, ensuring that forecast
models are tailored to location-specific weather patterns and historical trends. Each
model is trained on the complete historical dataset available for its respective location.

Model Training Features:
- Location-specific model training for improved forecast accuracy
- Automatic data retrieval and preprocessing from WeeklyWeatherHistory
- Robust error handling for individual model failures without stopping the entire process
- Standardized model persistence to the models directory

Dependencies:
- WeatherDatabase for historical data retrieval and management
- WeeklyForecastModel for machine learning model implementation
- WeeklyWeatherHistory ORM model for data access

Usage:
    This module is designed to run periodically to retrain forecast models with
    updated historical data. Models should be rebuilt when significant new historical
    data becomes available or when model performance degrades.

Example:
    python build_models.py

Scheduling:
    Recommended to run monthly depending on data volume and model performance
    requirements. Should run after sufficient new historical data has been accumulated.

Output:
    Trained models are saved to the ./models directory in the current working directory.
    Each model is specific to a geographic location and ready for use by the forecast service.

Note:
    The model building process may take considerable time depending on the number of
    locations and the volume of historical data. Individual model failures are logged
    but do not stop the training of models for other locations.
"""

import logging
import os

from weather_models import WeatherDatabase, WeeklyForecastModel, WeeklyWeatherHistory

LOGLEVEL = os.getenv("LOGLEVEL", "INFO")

if __name__ == "__main__":
    logging.basicConfig(
        level=LOGLEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Forecast Build Service")

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

                logger.info("Weekly forecast models build successfully!")
            except Exception as e:
                logger.exception(
                    f"An error occurred building the model for {location}:"
                )

    except Exception as e:
        logger.exception("An error occurred in the model build service: ")
