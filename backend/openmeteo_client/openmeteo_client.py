import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from datetime import date, datetime, timedelta
from time import sleep
from typing import Any, Dict, List, Literal, Tuple, Type

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from numpy.typing import NDArray
from openmeteo_sdk.VariablesWithTime import VariablesWithTime
from openmeteo_sdk.VariableWithValues import VariableWithValues
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse
from retry_requests import retry

from models import WeatherDatabase


@dataclass
class OpenMeteoClientConfig:
    history_start_date: date = field(init=False)
    history_end_date: date = field(init=False)
    forecast_days: int = field(init=False)
    forecast_past_days: int = field(init=False)
    locations: NDArray = field(init=False)
    metrics: List[str] = field(init=False)
    create_from_file: InitVar[bool] = field(default=False)
    config_file: InitVar[str | None] = field(default=None)

    def __post_init__(
        self, create_from_file: bool, config_file: str | None, **kwargs: Any
    ):
        """Initializes an OpenMeteoClientConfig from a config file or kwargs. Kwargs overwrite parameters from config file if set.

        Config parameters are:
            - history_start_date: Start date of historical weather data. Must be before history_end_date
            - history_end_date: End date of historical weather data. Must be at least two days before the current date. If "latest", defaults to two days before the current date.
            - forecast_days: Number of days to retrieve forecast for. Must be between 1 and 16 (incl.)
            - forecast_past_days: Currently non-functional. Must be 1.
            - bounding_box: Outer most coordinates to create a coordinate grid within. Used for computing locations to get weather data for. Must be of scheme: {"north": float, "south": float, "west": float, "east": float}.
            - metrics: List of metrics to retrieve. Can be any daily metric supported by Open Meteo Historic and Forecast APIs.

        Kwargs are required when no config file is given.

        Config file needs to be in json format and of the following schema:

        {
            "history_start_date": "YYYY-MM-DD",
            "history_end_date": "latest" | "YYYY-MM-DD",
            "forecast_days": int,
            "forecast_past_days": int,
            "bounding_box": {
                "north": float,
                "south": float,
                "west": float,
                "east": float
            },
            "metrics": [
                "daily_metric",
                "daily_metric",
            ]
        }

        Args:
            create_from_file (bool): Indicates whether to load config parameters from a file.
            config_file (str | None): Path to config file. Required when create_from_file=True.

        Kwargs:
            history_start_date (str | date)
            history_end_date (str | date)
            forecast_days (int)
            forecast_past_days (int),
            bounding_box (Dict[str, float])
            metrics (List[str])

        Raises:
            ValueError: When config_file is not given, but create_from_file=True.
        """
        if create_from_file:
            if config_file:
                config = self.__get_config(config_file)

                self.__set_history_start_date(config.get("history_start_date"))
                self.__set_history_end_date(config.get("history_end_date"))
                self.__set_forecast_days(config.get("forecast_days"))
                self.__set_forecast_past_days(config.get("forecast_past_days"))
                self.__set_locations(config.get("bounding_box"))
                self.__set_metrics(config.get("metrics"))

                self.__overwrite_kwargs(kwargs)
            else:
                raise ValueError(
                    "Parameter config_file is required, when create_from_file=True"
                )

        else:
            self.__set_history_start_date(kwargs.get("history_start_date"))
            self.__set_history_end_date(kwargs.get("history_end_date"))
            self.__set_forecast_days(kwargs.get("forecast_days"))
            self.__set_forecast_past_days(kwargs.get("forecast_past_days"))
            self.__set_locations(kwargs.get("bounding_box"))
            self.__set_metrics(kwargs.get("metrics"))

    def __get_config(self, config_file: str) -> Dict[str, Any]:
        """Load config from file.

        Args:
            config_file (str): Path to the config file.

        Returns:
            Dict: Config.
        """
        with open(file=config_file, mode="r") as file:
            config = json.load(fp=file)
            file.close()

        return config

    def __parse_date(self, date_string: str) -> date:
        """Parses a date from a string.

        Args:
            date str: Date in string format.

        Returns:
            date: Parsed datetime.date object.
        """
        return datetime.strptime(date_string, "%Y-%m-%d").date()

    def __compute_end_date(self) -> date:
        """Computes the end date as the date two days prior if end date is configured as latest.

        The Open Meteo Historical API provides data with a delay of two days.

        Returns:
            date: Computed datetime.date object.
        """
        return date.today() - timedelta(days=2)

    def __create_locations(self, bounding_box: Any, step: float = 0.5) -> NDArray:
        """Creates a matrix representing a grid of Lat. and Long. coordinates inside a bounding box.

        Args:
            bounding_box Any: Bounding box to create the grid in. Represented as extreme points of the coordinate grid.
            step (float, optional): Lat. and Long. steps to take. Defaults to 0.5.

        Raises:
            ValueError: When bounding_box does not match: {Dict[str, float]} with schema: {dict(north=float, south=float, east=float, west=float)}.

        Returns:
            NDArray: Matrix of Lat. and Long. coordinates.
        """
        if (
            isinstance(bounding_box, dict)
            and isinstance(bounding_box["south"], float)
            and isinstance(bounding_box["north"], float)
            and isinstance(bounding_box["west"], float)
            and isinstance(bounding_box["east"], float)
        ):
            latitude_range = np.arange(
                bounding_box["south"],
                bounding_box["north"],
                step,
            )

            longitude_range = np.arange(
                bounding_box["west"], bounding_box["east"], step
            )

            lat_grid, lon_grid = np.meshgrid(
                latitude_range, longitude_range, indexing="ij"
            )

            locations = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

            return locations
        else:
            raise ValueError(
                f"Kwarg bounding_box is required when create_from_file=False. Expected {Dict[str, float]} with schema: {dict(north=float, south=float, east=float, west=float)} Instead received: {bounding_box}"
            )

    def __set_history_start_date(self, history_start_date: Any) -> None:
        """Validates and sets the history_start_date attribute.

        Args:
            history_start_date (Any): Given value for history_start_date kwarg.

        Raises:
            ValueError: When history_start_date is not of expected type.
        """
        if isinstance(history_start_date, str):
            self.history_start_date = self.__parse_date(history_start_date)
        elif isinstance(history_start_date, date):
            self.history_start_date = history_start_date
        else:
            raise ValueError(
                f"Kwarg history_start_date is required when create_from_file=False. Expected {type(OpenMeteoClientConfig.history_start_date)} Received {type(history_start_date)} instead."
            )

    def __set_history_end_date(self, history_end_date: Any) -> None:
        """Validates and sets the history_end_date attribute.

        Args:
            history_end_date (Any): Given value for history_end_date kwarg.

        Raises:
            ValueError: When history_end_date is not of expected type.
        """
        if isinstance(history_end_date, str):
            if history_end_date == "latest":
                self.history_end_date = self.__compute_end_date()
            else:
                self.history_end_date = self.__parse_date(history_end_date)
        elif isinstance(history_end_date, date):
            self.history_end_date = history_end_date
        else:
            raise ValueError(
                f"Kwarg history_end_date is required when create_from_file=False. Expected {type(OpenMeteoClientConfig.history_end_date)} Received {type(history_end_date)} instead."
            )

    def __set_forecast_days(self, forecast_days: Any) -> None:
        """Validates and sets the forecast_days attribute.

        Args:
            forecast_days (Any): Given value for forecast_days kwarg.

        Raises:
            ValueError: When forecast_days not >0.
            ValueError: When forecast_days is not of expected type.
        """
        if isinstance(forecast_days, int):
            if forecast_days > 0:
                self.forecast_days = forecast_days
            else:
                raise ValueError(
                    f"Parameter forecast_days must be >0. Got {forecast_days}"
                )
        else:
            raise ValueError(
                f"Kwarg forecast_days is required when create_from_file=False. Expected {type(OpenMeteoClientConfig.forecast_days)} Received {type(forecast_days)} instead."
            )

    def __set_forecast_past_days(self, forecast_past_days: Any) -> None:
        """Validates and sets the forecast_past_days attribute.

        Args:
            forecast_past_days (Any): Given value for forecast_past_days kwarg.

        Raises:
            ValueError: When forecast_past_days between 1 and 5 (incl.)
            ValueError: When forecast_past_days is not of expected type.
        """
        if isinstance(forecast_past_days, int):
            if 5 >= forecast_past_days > 0:
                self.forecast_past_days = forecast_past_days
            else:
                raise ValueError(
                    f"Parameter forecast_past_days must be between 1 and 5(incl.) Got {forecast_past_days}"
                )
        else:
            raise ValueError(
                f"Kwarg forecast_past_days is required when create_from_file=False. Expected {type(OpenMeteoClientConfig.forecast_past_days)} Received {type(forecast_past_days)} instead."
            )

    def __set_locations(self, bounding_box: Any) -> None:
        """Validates and sets the locations attribute.

        Args:
            bounding_box (Any): Bounding box to create the grid in. Represented as extreme points of the coordinate grid.
        """
        self.locations = self.__create_locations(bounding_box)

    def __set_metrics(self, metrics: Any) -> None:
        """Validates and sets the metrics attribute.

        Args:
            metrics (Any): Given value for metrics kwarg.

        Raises:
            ValueError: When metrics is not of expected type.
        """
        if isinstance(metrics, list):
            self.metrics = [str(metric) for metric in metrics]
        else:
            raise ValueError(
                f"Kwarg metrics is required when create_from_file=False. Expected {type(OpenMeteoClientConfig.metrics)} Received {type(metrics)} instead."
            )

    def __overwrite_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Function to have kwargs overwrite config from file if set.

        Args:
            kwargs (Dict[str, Any]): kwargs passed to constructor.
        """
        if kwargs.get("history_start_date"):
            self.__set_history_start_date(kwargs.get("history_start_date"))
        if kwargs.get("history_end_date"):
            self.__set_history_end_date(kwargs.get("history_end_date"))
        if kwargs.get("forecast_days"):
            self.__set_forecast_days(kwargs.get("forecast_days"))
        if kwargs.get("forecast_past_days"):
            self.__set_forecast_past_days(kwargs.get("forecast_past_days"))
        if kwargs.get("bounding_box"):
            self.__set_locations(kwargs.get("bounding_box"))
        if kwargs.get("metrics"):
            self.__set_metrics(kwargs.get("metrics"))


class OpenMeteoClient(ABC, openmeteo_requests.Client):

    SESSION = retry(
        requests_cache.CachedSession(".cache", expire_after=86399),
        retries=10,
        backoff_factor=2,
    )

    MINUTELY_RATE_LIMIT = 600
    HOURLY_RATE_LIMIT = 5000
    DAILY_RATE_LIMIT = 10000
    MINUTELY_BACKOFF = 61
    HOURLY_BACKOFF = 3601
    DAILY_BACKOFF = 86401

    def __init__(self, create_from_file: bool):
        """_summary_

        Args:
            session (Session, optional): _description_. Defaults to SESSION.
        """
        super().__init__(OpenMeteoClient.SESSION)  # type: ignore

        self.config = OpenMeteoClientConfig(
            create_from_file=create_from_file,
            config_file=os.path.join(
                os.path.dirname(__file__), os.getenv("CONFIG_FILE", "config.json")
            ),
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.logger.info(f"Setting up {self.__class__.__name__}")

        self.database = WeatherDatabase()

    @abstractmethod
    def get_data(self, url: str) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        pass

    def get_request_time_estimate(self, num_requests: int) -> float:
        if num_requests <= OpenMeteoClient.MINUTELY_RATE_LIMIT:
            time_estimate = 0.0
        elif (
            OpenMeteoClient.MINUTELY_RATE_LIMIT
            < num_requests
            <= OpenMeteoClient.HOURLY_RATE_LIMIT
        ):
            time_estimate = (
                int(
                    (num_requests - OpenMeteoClient.MINUTELY_RATE_LIMIT)
                    / OpenMeteoClient.MINUTELY_RATE_LIMIT
                )
                * OpenMeteoClient.MINUTELY_BACKOFF
            )
        elif (
            OpenMeteoClient.HOURLY_RATE_LIMIT
            < num_requests
            <= OpenMeteoClient.DAILY_RATE_LIMIT
        ):
            time_estimate = (
                int(
                    (num_requests - OpenMeteoClient.HOURLY_RATE_LIMIT)
                    / OpenMeteoClient.HOURLY_RATE_LIMIT
                )
                * OpenMeteoClient.HOURLY_BACKOFF
            )
        elif OpenMeteoClient.DAILY_RATE_LIMIT < num_requests:
            time_estimate = (
                int(
                    (num_requests - OpenMeteoClient.DAILY_RATE_LIMIT)
                    / OpenMeteoClient.DAILY_RATE_LIMIT
                )
                * OpenMeteoClient.DAILY_BACKOFF
            )
        else:
            time_estimate = 0.0

        return time_estimate

    def handle_ratelimit(
        self,
        minutely_usage: float,
        hourly_usage: float,
        daily_usage: float,
        fractional_api_cost: float,
    ) -> Tuple[float, float, float]:
        if minutely_usage + fractional_api_cost >= OpenMeteoClient.MINUTELY_RATE_LIMIT:
            self.logger.info(
                f"Minutely rate limit hit. Backing off for {str(timedelta(seconds=OpenMeteoClient.MINUTELY_BACKOFF))}."
            )
            sleep(OpenMeteoClient.MINUTELY_BACKOFF)
            minutely_usage = 0.0
        if hourly_usage + fractional_api_cost >= OpenMeteoClient.HOURLY_RATE_LIMIT:
            self.logger.info(
                f"Hourly rate limit hit. Backing off for {str(timedelta(seconds=OpenMeteoClient.HOURLY_BACKOFF))}."
            )
            sleep(OpenMeteoClient.HOURLY_BACKOFF)
            minutely_usage = 0.0
            hourly_usage = 0.0
        if daily_usage + fractional_api_cost >= OpenMeteoClient.DAILY_RATE_LIMIT:
            self.logger.info(
                f"Daily rate limit hit. Backing off for {str(timedelta(seconds=OpenMeteoClient.DAILY_BACKOFF))} seconds."
            )
            sleep(OpenMeteoClient.DAILY_BACKOFF)
            minutely_usage = 0.0
            hourly_usage = 0.0
            daily_usage = 0.0

        return (minutely_usage, hourly_usage, daily_usage)

    def extract_variable(
        self, variable_index: int, variables: VariablesWithTime
    ) -> np.ndarray:
        """_summary_

        Args:
            variable_index (int): _description_
            variables (VariablesWithTime): _description_

        Raises:
            TypeError: _description_

        Returns:
            np.ndarray: _description_
        """
        variable = variables.Variables(variable_index)

        if isinstance(variable, VariableWithValues):
            values = variable.ValuesAsNumpy()
        else:
            raise TypeError(
                f"Error during variable extraction. Expected type: {VariableWithValues} Got: {type(variable)} instead."
            )

        return values

    def process_response(
        self, response: WeatherApiResponse, config: OpenMeteoClientConfig
    ) -> pd.DataFrame:
        """_summary_

        Args:
            response (WeatherApiResponse): _description_
            params (Dict[str, Any]): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        daily = response.Daily()

        if isinstance(daily, VariablesWithTime):
            variables = [
                self.extract_variable(idx, daily) for idx in range(len(config.metrics))
            ]

            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left",
                )
            }

            for idx, variable_name in enumerate(config.metrics):
                daily_data[variable_name] = variables[idx].tolist()

        else:
            raise TypeError(
                f"Error during processing response. Expected type: {VariablesWithTime} Got: {type(daily)} instead."
            )

        return pd.DataFrame(data=daily_data)

    def _main(self, url: str) -> pd.DataFrame:
        """_summary_

        Args:
            url (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = pd.DataFrame()
        for response in self.get_data(url):
            processed_response = self.process_response(
                response=response, config=self.config
            )

            processed_response["latitude"] = response.Latitude()
            processed_response["longitude"] = response.Longitude()

            data = pd.concat([data, processed_response], axis=0)

        data["date"] = pd.to_datetime(
            data["date"], format="%Y-%m-%d %H:%M:%S"
        ).dt.strftime("%Y-%m-%d")

        self.logger.info(f"{self.__class__.__name__} exited successfully.")

        return data


class OpenMeteoArchiveClient(OpenMeteoClient):

    URL = "https://archive-api.open-meteo.com/v1/archive"

    FRACTIONAL_API_COST = 31.3

    def get_data(self, url: str) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        responses = []

        years = list(
            range(
                self.config.history_start_date.year,
                self.config.history_end_date.year + 1,
            )
        )

        num_requests = self.config.locations.shape[0] * len(years)
        time_estimate = self.get_request_time_estimate(num_requests)

        self.logger.info(
            f"Processing {num_requests} requests costing an estimated {OpenMeteoArchiveClient.FRACTIONAL_API_COST * num_requests} API calls.\nThis will take ~ {str(timedelta(seconds=time_estimate))}"
        )

        minutely_usage = 0.0
        hourly_usage = 0.0
        daily_usage = 0.0

        for location in self.config.locations:
            for year in years:
                start_date = date(year, 1, 1)
                end_date = (
                    date(year, 12, 31)
                    if year < date.today().year
                    else self.config.history_end_date
                )
                fractional_query_params = {
                    "latitude": location[0],
                    "longitude": location[1],
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": self.config.metrics,
                }

                self.logger.info(
                    f"Retrieving historic data for Lat.: {location[0]}° (N), Lon.: {location[1]}° (E) from {start_date} to {end_date}"
                )

                fractional_responses = self.weather_api(
                    url, params=fractional_query_params
                )

                responses.append(*fractional_responses)

                minutely_usage, hourly_usage, daily_usage = (
                    minutely_usage + OpenMeteoArchiveClient.FRACTIONAL_API_COST,
                    hourly_usage + OpenMeteoArchiveClient.FRACTIONAL_API_COST,
                    daily_usage + OpenMeteoArchiveClient.FRACTIONAL_API_COST,
                )

                minutely_usage, hourly_usage, daily_usage = self.handle_ratelimit(
                    minutely_usage,
                    hourly_usage,
                    daily_usage,
                    OpenMeteoArchiveClient.FRACTIONAL_API_COST,
                )

        return responses

    def main(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        data = self._main(url=OpenMeteoArchiveClient.URL)

        return data


class OpenMeteoForecastClient(OpenMeteoClient):

    URL = "https://api.open-meteo.com/v1/forecast"

    FRACTIONAL_API_COST = 1.2

    def __init__(self, create_from_file: bool):
        super().__init__(create_from_file)

        self.start_date = date.today() - timedelta(days=self.config.forecast_past_days)
        self.end_date = (
            date.today() + timedelta(days=self.config.forecast_days) - timedelta(days=1)
        )

    def get_data(self, url: str) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        responses = []

        num_requests = self.config.locations.shape[0]
        time_estimate = self.get_request_time_estimate(num_requests)

        self.logger.info(
            f"Processing {num_requests} requests costing an estimated {OpenMeteoForecastClient.FRACTIONAL_API_COST * num_requests} API calls.\nThis will take ~ {str(timedelta(seconds=time_estimate))}"
        )

        minutely_usage = 0.0
        hourly_usage = 0.0
        daily_usage = 0.0

        for location in self.config.locations:
            fractional_query_params = {
                "latitude": location[0],
                "longitude": location[1],
                "past_days": self.config.forecast_past_days,
                "forecast_days": self.config.forecast_days,
                "daily": self.config.metrics,
            }

            self.logger.info(
                f"Retrieving forecast data for Lat.: {location[0]}° (N), Lon.: {location[1]}° (E)"
            )

            fractional_responses = self.weather_api(url, params=fractional_query_params)

            responses.append(*fractional_responses)

            minutely_usage, hourly_usage, daily_usage = (
                minutely_usage + OpenMeteoForecastClient.FRACTIONAL_API_COST,
                hourly_usage + OpenMeteoForecastClient.FRACTIONAL_API_COST,
                daily_usage + OpenMeteoForecastClient.FRACTIONAL_API_COST,
            )

            self.handle_ratelimit(
                minutely_usage,
                hourly_usage,
                daily_usage,
                OpenMeteoForecastClient.FRACTIONAL_API_COST,
            )

        return responses

    def main(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        data = self._main(url=OpenMeteoForecastClient.URL)

        return data


class WeeklyTableConstructor:

    def __trim_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Trims data to full weeks. Cutoffs are first Monday in the front and last Sunday in the back.

        Args:
            data (pd.DataFrame): Daily data to trim.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of trimmed DataFrames. (main, head, tail)
        """
        cutoff_front = data[data["date"].dt.weekday == 0]["date"].min()
        cutoff_back = data[data["date"].dt.weekday == 6]["date"].max()

        data_main = (
            data[(data["date"] >= cutoff_front) & (data["date"] <= cutoff_back)]
        ).copy()

        data_head = (data[data["date"] < cutoff_front]).copy()

        data_tail = (data[data["date"] > cutoff_back]).copy()

        return data_main, data_head, data_tail

    def __aggregate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregates data by week and location.

        Args:
            data (pd.DataFrame): Weekly DataFrame to aggregate.

        Returns:
            pd.DataFrame: Aggregated DataFrame.
        """
        data["week"] = data["date"] + pd.offsets.Week(weekday=6)
        data["year"] = data["week"].dt.year
        data = data.drop(columns="date")

        data = (
            data.groupby(["week", "year", "latitude", "longitude"])
            .aggregate(
                {
                    "temperature_2m_mean": "mean",
                    "temperature_2m_max": "max",
                    "temperature_2m_min": "min",
                    "cloud_cover_mean": "mean",
                    "cloud_cover_max": "max",
                    "cloud_cover_min": "min",
                    "wind_speed_10m_mean": "mean",
                    "wind_speed_10m_min": "min",
                    "wind_speed_10m_max": "max",
                    "sunshine_duration": "mean",
                    "precipitation_sum": "sum",
                    "precipitation_hours": "sum",
                }
            )
            .reset_index()
        )

        return data

    def __create_calendar_week(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates a calendar week column.

        Args:
            data (pd.DataFrame): Weekly DataFrame.

        Returns:
            pd.DataFrame: Resulting DataFrame.
        """
        data["week"] = data["week"].dt.isocalendar().week

        return data

    def main(
        self, daily_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Main routine of WeeklyTableConstructor class. Creates a weekly DataFrame from daily weather data. Returns the result DataFrame as well as cutoff data.

        Args:
            daily_data (pd.DataFrame): Daily weather data to aggregate.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of DataFrames. (main, head, tail)
        """
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        data_main, data_head, data_tail = self.__trim_data(data=daily_data)
        data_main = self.__aggregate_data(data_main)
        data_main = self.__create_calendar_week(data_main)

        return data_main, data_head, data_tail
