import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from time import sleep
from typing import Any, Dict, List, Tuple

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from niquests import Session
from numpy.typing import NDArray
from openmeteo_sdk.VariablesWithTime import VariablesWithTime
from openmeteo_sdk.VariableWithValues import VariableWithValues
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse
from retry_requests import retry
from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import Session, sessionmaker

from models import DailyWeatherForecast, DailyWeatherHistory, DatabaseEngine


class OpenMeteoClient(ABC, openmeteo_requests.Client):

    SESSION = retry(
        requests_cache.CachedSession(".cache", expire_after=86399),
        retries=10,
        backoff_factor=2,
    )

    CONFIG_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )

    MINUTELY_RATE_LIMIT = 600
    HOURLY_RATE_LIMIT = 5000
    DAILY_RATE_LIMIT = 10000
    MINUTELY_BACKOFF = 61
    HOURLY_BACKOFF = 3601
    DAILY_BACKOFF = 86401

    def __init__(self, session: Session = SESSION):  # type: ignore
        """_summary_

        Args:
            session (Session, optional): _description_. Defaults to SESSION.
        """
        super().__init__(session)  # type: ignore

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)

    @abstractmethod
    def check_data_exists(self) -> bool:
        """_summary_

        Args:
            table (str): _description_

        Returns:
            bool: _description_
        """
        pass

    @abstractmethod
    def get_data(self, url: str, params: Dict[str, Any]) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        pass

    def get_request_time_estimate(self, num_requests: int) -> float:  # TODO: fix
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
        self, response: WeatherApiResponse, params: Dict[str, Any]
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
                self.extract_variable(idx, daily) for idx in range(len(params["daily"]))
            ]

            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left",
                )
            }

            for idx, variable_name in enumerate(params["daily"]):
                daily_data[variable_name] = variables[idx].tolist()

        else:
            raise TypeError(
                f"Error during processing response. Expected type: {VariablesWithTime} Got: {type(daily)} instead."
            )

        return pd.DataFrame(data=daily_data)

    @abstractmethod
    def main(self) -> None:
        """_summary_

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        pass


class OpenMeteoArchiveClient(OpenMeteoClient):

    URL = "https://archive-api.open-meteo.com/v1/archive"

    FRACTIONAL_API_COST = 31.3

    def __init__(self, session: Session = OpenMeteoClient.SESSION):  # type: ignore
        """_summary_

        Args:
            session (Session, optional): _description_. Defaults to SESSION.
        """
        super().__init__(session)

        self.logger.info(f"Setting up {self.__class__.__name__}")

        self.DB_SESSION = sessionmaker(bind=DatabaseEngine().get_engine)()

        with open(file=OpenMeteoClient.CONFIG_FILE, mode="r") as file:
            config = json.load(fp=file)
            file.close()

        latitude_range = np.arange(
            config["bounding_box"]["south_boundary"],
            config["bounding_box"]["north_boundary"],
            0.5,
        )
        longitude_range = np.arange(
            config["bounding_box"]["west_boundary"],
            config["bounding_box"]["east_boundary"],
            0.5,
        )
        lat_grid, lon_grid = np.meshgrid(latitude_range, longitude_range, indexing="ij")
        locations = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        self.QUERY_PARAMS = {
            "locations": locations,
            "start_date": config["history_start_date"],
            "end_date": (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"),
            "daily": config["metrics_daily"],
        }

    def check_data_exists(self) -> bool:
        """_summary_

        Args:
            table (str): _description_

        Returns:
            bool: _description_
        """
        self.logger.info("Checking if historic data already exists as expected...")

        expected_start_date = datetime.strptime(
            self.QUERY_PARAMS["start_date"], "%Y-%m-%d"
        ).date()
        expected_end_date = datetime.strptime(
            self.QUERY_PARAMS["end_date"], "%Y-%m-%d"
        ).date()
        start_date = self.DB_SESSION.scalar(select(func.min(DailyWeatherHistory.date)))
        end_date = self.DB_SESSION.scalar(select(func.max(DailyWeatherHistory.date)))
        date_range = self.DB_SESSION.scalars(
            select(distinct(DailyWeatherHistory.date))
        ).all()

        if isinstance(start_date, date) and isinstance(end_date, date):
            if (
                start_date == expected_start_date
                and end_date == expected_end_date
                and len(date_range) == (end_date - start_date).days + 1
            ):
                self.logger.info(
                    "Historic data exists as expected. Skipping data retrieval..."
                )
                return True
            else:
                self.logger.info(
                    f"Schema:\nStart Date : {start_date}, End Date: {end_date}, Date Range: {len(date_range)} days\nDoes not match expected schema:\nStart Date : {expected_start_date}, End Date: {expected_end_date}, Date Range: {(end_date - start_date).days + 1} days"
                )
                return False
        else:
            self.logger.info(
                f"Data does not match expected type:\nStart Date: {start_date} Type: {type(start_date)}; End Date: {end_date} Type: {type(end_date)}"
            )
            return False

    def get_data(self, url: str, params: Dict[str, Any]) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        responses = []

        locations: NDArray[np.float64] = params["locations"]

        years = list(
            range(
                datetime.strptime(params["start_date"], "%Y-%m-%d").date().year,
                datetime.strptime(params["end_date"], "%Y-%m-%d").date().year + 1,
            )
        )

        num_requests = locations.shape[0] * len(years)
        time_estimate = self.get_request_time_estimate(num_requests)

        self.logger.info(
            f"Processing {num_requests} requests costing an estimated {OpenMeteoArchiveClient.FRACTIONAL_API_COST * num_requests} API calls.\nThis will take ~ {str(timedelta(seconds=time_estimate))}"
        )

        minutely_usage = 0.0
        hourly_usage = 0.0
        daily_usage = 0.0

        for location in locations:
            for year in years:
                start_date = date(year, 1, 1)
                end_date = (
                    date(year, 12, 31)
                    if year < date.today().year
                    else params["end_date"]
                )
                fractional_query_params = {
                    "latitude": location[0],
                    "longitude": location[1],
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": params["daily"],
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

    def main(self) -> None:
        """_summary_"""
        if not self.check_data_exists():
            self.logger.info("Cleaning table...")
            self.DB_SESSION.execute(delete(DailyWeatherHistory))
            data = pd.DataFrame()
            for response in self.get_data(
                url=OpenMeteoArchiveClient.URL, params=self.QUERY_PARAMS
            ):
                processed_response = self.process_response(
                    response=response, params=self.QUERY_PARAMS
                )

                processed_response["latitude"] = response.Latitude()
                processed_response["longitude"] = response.Longitude()

                data = pd.concat([data, processed_response], axis=0)

            data["date"] = pd.to_datetime(
                data["date"], format="%Y-%m-%d %H:%M:%S"
            ).dt.strftime("%Y-%m-%d")

            self.logger.info("Writing data to database...")
            records = data.to_dict(orient="records")
            orm_objects = [
                DailyWeatherHistory(**{str(k): v for k, v in row.items()})
                for row in records
            ]

            self.DB_SESSION.add_all(orm_objects)
            self.DB_SESSION.commit()

        self.DB_SESSION.close()

        self.logger.info(f"{self.__class__.__name__} exited successfully.")


class OpenMeteoForecastClient(OpenMeteoClient):

    URL = "https://api.open-meteo.com/v1/forecast"

    FRACTIONAL_API_COST = 1.2

    def __init__(self, session: Session = OpenMeteoClient.SESSION):  # type: ignore
        """_summary_

        Args:
            session (Session, optional): _description_. Defaults to SESSION.
        """
        super().__init__(session)

        self.logger.info(f"Setting up {self.__class__.__name__}")

        self.DB_SESSION = sessionmaker(bind=DatabaseEngine().get_engine)()

        with open(file=OpenMeteoClient.CONFIG_FILE, mode="r") as file:
            config = json.load(fp=file)
            file.close()

        latitude_range = np.arange(
            config["bounding_box"]["south_boundary"],
            config["bounding_box"]["north_boundary"],
            0.5,
        )
        longitude_range = np.arange(
            config["bounding_box"]["west_boundary"],
            config["bounding_box"]["east_boundary"],
            0.5,
        )
        lat_grid, lon_grid = np.meshgrid(latitude_range, longitude_range, indexing="ij")
        locations = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        self.QUERY_PARAMS = {
            "locations": locations,
            "past_days": config["forecast_past_days"],
            "forecast_days": config["forecast_days"],
            "daily": config["metrics_daily"],
        }

    def check_data_exists(self) -> bool:
        """_summary_

        Args:
            table (str): _description_

        Returns:
            bool: _description_
        """
        self.logger.info("Checking if forecast data already exists as expected...")

        expected_start_date = date.today() - timedelta(
            days=self.QUERY_PARAMS["past_days"]
        )
        expected_end_date = date.today() + timedelta(
            days=self.QUERY_PARAMS["forecast_days"]
        )
        start_date = self.DB_SESSION.scalar(select(func.min(DailyWeatherForecast.date)))
        end_date = self.DB_SESSION.scalar(select(func.max(DailyWeatherForecast.date)))
        date_range = self.DB_SESSION.scalars(
            select(distinct(DailyWeatherForecast.date))
        ).all()

        if isinstance(start_date, date) and isinstance(end_date, date):
            if (
                start_date == expected_start_date
                and end_date == expected_end_date
                and len(date_range) == (end_date - start_date).days + 1
            ):
                self.logger.info(
                    "Forecast data exists as expected. Skipping data retrieval..."
                )
                return True
            else:
                self.logger.info(
                    f"Schema:\nStart Date : {start_date}, End Date: {end_date}, Date Range: {len(date_range)} days\nDoes not match expected schema:\nStart Date : {expected_start_date}, End Date: {expected_end_date}, Date Range: {(end_date - start_date).days + 1} days"
                )
                return False
        else:
            self.logger.info(
                f"Data does not match expected type:\nStart Date: {start_date} Type: {type(start_date)}; End Date: {end_date} Type: {type(end_date)}"
            )
            return False

    def get_data(self, url: str, params: Dict[str, Any]) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        responses = []

        locations: NDArray[np.float64] = params["locations"]

        num_requests = locations.shape[0]
        time_estimate = self.get_request_time_estimate(num_requests)

        self.logger.info(
            f"Processing {num_requests} requests costing an estimated {OpenMeteoForecastClient.FRACTIONAL_API_COST * num_requests} API calls.\nThis will take ~ {str(timedelta(seconds=time_estimate))}"
        )

        minutely_usage = 0.0
        hourly_usage = 0.0
        daily_usage = 0.0

        for location in locations:
            fractional_query_params = {
                "latitude": location[0],
                "longitude": location[1],
                "past_days": self.QUERY_PARAMS["past_days"],
                "forecast_days": self.QUERY_PARAMS["forecast_days"],
                "daily": params["daily"],
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

    def main(self) -> None:
        """_summary_"""
        if not self.check_data_exists():
            self.logger.info("Cleaning table...")
            self.DB_SESSION.execute(delete(DailyWeatherForecast))
            data = pd.DataFrame()
            for response in self.get_data(
                url=OpenMeteoForecastClient.URL, params=self.QUERY_PARAMS
            ):
                processed_response = self.process_response(
                    response=response, params=self.QUERY_PARAMS
                )

                processed_response["latitude"] = response.Latitude()
                processed_response["longitude"] = response.Longitude()

                data = pd.concat([data, processed_response], axis=0)

            data["date"] = pd.to_datetime(
                data["date"], format="%Y-%m-%d %H:%M:%S"
            ).dt.strftime("%Y-%m-%d")

            self.logger.info("Writing data to database...")
            records = data.to_dict(orient="records")
            orm_objects = [
                DailyWeatherForecast(**{str(k): v for k, v in row.items()})
                for row in records
            ]

            self.DB_SESSION.add_all(orm_objects)
            self.DB_SESSION.commit()

        self.DB_SESSION.close()

        self.logger.info(f"{self.__class__.__name__} exited successfully.")
