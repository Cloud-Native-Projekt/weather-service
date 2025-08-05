import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from niquests import Session
from openmeteo_sdk.VariablesWithTime import VariablesWithTime
from openmeteo_sdk.VariableWithValues import VariableWithValues
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse
from retry_requests import retry


class OpenMeteoClient(ABC, openmeteo_requests.Client):

    SESSION = retry(
        requests_cache.CachedSession(".cache", expire_after=-1),
        retries=5,
        backoff_factor=0.2,
    )

    CONFIG_FILE = os.path.join(os.getcwd(), "config.json")

    def __init__(self, session: Session = SESSION):  # type: ignore
        """_summary_

        Args:
            session (Session, optional): _description_. Defaults to SESSION.
        """
        super().__init__(session)

    def get_data(self, url: str, params: Dict[str, Any]) -> List[WeatherApiResponse]:
        """_summary_

        Args:
            url (str): URL of the API endpoint
            params (Dict[str, Any]): Dictionary containing the query params of the API request

        Returns:
            List[WeatherApiResponse]: List of responses from the OpenMeteo Weather API
        """
        responses = self.weather_api(url, params=params)

        return responses

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

    def __init__(self, session: Session = OpenMeteoClient.SESSION):  # type: ignore
        """_summary_

        Args:
            session (Session, optional): _description_. Defaults to SESSION.
        """
        super().__init__(session)

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
        coordinate_matrix = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        latitude_list = list(coordinate_matrix[:, 0])
        longitude_list = list(coordinate_matrix[:, 1])

        self.QUERY_PARAMS = {
            "latitude": latitude_list,
            "longitude": longitude_list,
            "start_date": config["history_start_date"],
            "end_date": (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"),
            "daily": config["metrics_daily"],
        }

    def main(self) -> None:
        """_summary_"""
        data = pd.DataFrame()
        for response in self.get_data(
            url=OpenMeteoArchiveClient.URL, params=self.QUERY_PARAMS
        ):
            regional_data = self.process_response(
                response=response, params=self.QUERY_PARAMS
            )

            regional_data["latitude"] = response.Latitude()
            regional_data["longitude"] = response.Longitude()

            data = pd.concat([data, regional_data], axis=0)

        print(data)


if __name__ == "__main__":
    client = OpenMeteoArchiveClient()
    client.main()
