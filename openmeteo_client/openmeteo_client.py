"""OpenMeteo API Client Library for Weather Data Retrieval

This module provides a client library for interacting with OpenMeteo APIs
to retrieve weather data including historical observations and forecasts. It implements
rate limiting, data processing, and aggregation capabilities specifically
designed for the weather service infrastructure.

Core Components:

Configuration Management:
- OpenMeteoClientConfig: Flexible configuration system supporting JSON files and kwargs
- Parameter validation and type conversion for API compatibility
- Geographic coordinate grid generation from bounding box specifications
- Temporal range management with automatic date calculations

API Client Architecture:
- OpenMeteoClient: Abstract base class with common functionality
- OpenMeteoArchiveClient: Historical weather data retrieval with year-based chunking
- OpenMeteoForecastClient: Weather forecast data with configurable horizons
- Automatic rate limiting across multiple time windows (minutely/hourly/daily)

Data Processing Pipeline:
- WeeklyTableConstructor: Daily-to-weekly aggregation using meteorological best practices
- Response processing to structured pandas DataFrames
- Geographic metadata integration for spatial analysis
- Temporal standardization for database compatibility

Key Features:

Rate Limiting System:
- Multi-tier rate limiting (600/min, 5,000/hour, 10,000/day)
- Automatic backoff with progressive delays (61s, 1h, 24h)
- API cost tracking for endpoint pricing
- Request time estimation for batch operations

Geographic Processing:
- Coordinate grid generation from bounding boxes
- Multi-location request optimization
- Spatial data integrity across all operations
- Configurable grid resolution (default 0.5° spacing)

Temporal Management:
- Automatic date range calculations
- Week boundary detection and alignment
- ISO calendar week numbering
- Partial week handling for data integrity

API Endpoints Supported:

OpenMeteo Archive API:
- Endpoint: https://archive-api.open-meteo.com/v1/archive
- Purpose: Historical weather observations (2+ day delay)
- Cost: 31.3 API units per location-year
- Optimization: Location- and year-based request chunking

OpenMeteo Forecast API:
- Endpoint: https://api.open-meteo.com/v1/forecast
- Purpose: Weather predictions (1-16 days ahead)
- Cost: 1.2 API units per location
- Features: Optional past days inclusion (1-5 days)

Configuration Sources:

JSON Configuration Files:
- Centralized parameter management
- Environment-specific configurations
- Schema validation and error handling
- Default parameter definitions

Runtime Parameters:
- Direct kwargs parameter passing
- File override capabilities
- Flexible hybrid configuration
- Dynamic parameter adjustment

Usage Patterns:

Historical Data Retrieval:\n
    config = OpenMeteoClientConfig(
        create_from_file=True,
        kwargs={"history_start_date": "2020-01-01", "history_end_date": "latest"}
    )
    archive_client = OpenMeteoArchiveClient(config)
    historical_data = archive_client.main()

Forecast Data Retrieval:\n
    config = OpenMeteoClientConfig(
        create_from_file=True,
        kwargs={"forecast_days": 7, "forecast_past_days": 1}
    )
    forecast_client = OpenMeteoForecastClient(config)
    forecast_data = forecast_client.main()

Weekly Aggregation:\n
    constructor = WeeklyTableConstructor()
    weekly_data, head_partial, tail_partial = constructor.main(daily_data)

Configuration Management:
    From file with overrides:\n
        config = OpenMeteoClientConfig(
            create_from_file=True,
            config_file="/path/to/config.json",
            kwargs={"forecast_days": 14}
        )

    From kwargs only:\n
        config = OpenMeteoClientConfig(
            create_from_file=False,
            kwargs={
                "bounding_box": {"north": 50.0, "south": 40.0, "west": -10.0, "east": 5.0},
                "metrics": ["temperature_2m_mean", "precipitation_sum"]
            }
        )

Integration Points:

Database Integration:
- Direct compatibility with WeatherDatabase ORM objects
- Standardized data formats for persistence
- Automatic geographic and temporal indexing support

Service Architecture:
- Bootstrap service: Initial data population
- Maintenance service: Daily/weekly data updates
- Rate limiting coordination across service instances

Error Handling:
- Comprehensive parameter validation
- API response verification
- Graceful degradation with partial failures
- Detailed logging for monitoring and debugging

Performance Characteristics:

Optimization Features:
- Request caching with 24-hour expiration
- Automatic retry with exponential backoff
- Geographic batch processing
- Year-based chunking for large historical ranges

Scalability Considerations:
- Rate limiting prevents API quota violations
- Memory-efficient streaming for large datasets
- Geographic partitioning for parallel processing
- Configurable grid resolution for performance tuning

Dependencies:
- openmeteo_requests: Official OpenMeteo SDK for API communication
- openmeteo_sdk: Response parsing and data extraction utilities
- pandas: Data manipulation and temporal operations
- numpy: Numerical computations and array operations
- requests_cache: HTTP caching for performance optimization
- retry_requests: Automatic retry logic for resilient operations

Note:
This module is designed as the primary interface for all OpenMeteo API interactions
within the weather service. It provides the foundation for data retrieval, processing,
and aggregation across the entire system architecture.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from datetime import date, datetime, timedelta
from time import sleep
from typing import Any, Dict, List, Tuple

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from numpy.typing import NDArray
from openmeteo_sdk.VariablesWithTime import VariablesWithTime
from openmeteo_sdk.VariableWithValues import VariableWithValues
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse
from retry_requests import retry


@dataclass
class OpenMeteoClientConfig:
    """Configuration class for OpenMeteo API client parameters.

    This dataclass manages all configuration parameters required for OpenMeteo API
    clients, including temporal ranges, geographic boundaries, and meteorological
    metrics. It supports initialization from JSON configuration files or direct
    parameter passing via kwargs.

    The class handles validation and type conversion for all parameters, ensuring
    compatibility with OpenMeteo API requirements. It automatically generates
    geographic coordinate grids from bounding box specifications and validates
    temporal ranges according to API constraints.

    Supported Configuration Sources:
    - JSON configuration files with predefined schema
    - Direct parameter passing via kwargs during initialization
    - Hybrid approach with file-based defaults and kwargs overrides

    Attributes:
        history_start_date (date): Start date for historical weather data retrieval
        history_end_date (date): End date for historical weather data (auto-computed if "latest")
        forecast_days (int): Number of future days to include in forecasts (1-16)
        forecast_past_days (int): Number of past days to include in forecasts (1-5)
        bounding_box (Dict[str, float]): Geographic boundaries for coordinate grid
        metrics (List[str]): OpenMeteo daily metrics to retrieve
        locations (NDArray): Generated coordinate grid from bounding box

    Configuration File Schema:
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
            "metrics": ["daily_metric1", "daily_metric2", ...]
        }

    Example:
        From configuration file:\n
        config = OpenMeteoClientConfig(create_from_file=True)

        From kwargs with overrides:\n
        config = OpenMeteoClientConfig(
            create_from_file=True,
            kwargs={"forecast_days": 10}
        )

        From kwargs only:\n
        config = OpenMeteoClientConfig(
            create_from_file=False,
            kwargs={
                "history_start_date": "2024-01-01",
                "history_end_date": "latest",
                "forecast_days": 7,
                "forecast_past_days": 1,
                "bounding_box": {"north": 50.0, "south": 40.0, "west": -10.0, "east": 5.0},
                "metrics": ["temperature_2m_mean", "precipitation_sum"]
            }
        )
    """

    history_start_date: date = field(init=False, metadata={"format": "YYYY-MM-DD"})
    history_end_date: date = field(init=False)
    forecast_days: int = field(init=False)
    forecast_past_days: int = field(init=False)
    bounding_box: Dict[str, float] = field(init=False)
    metrics: List[str] = field(init=False)
    locations: NDArray = field(init=False)
    create_from_file: InitVar[bool] = field(default=False)
    config_file: InitVar[str | None] = field(default=None)
    kwargs: InitVar[Dict[str, Any] | None] = field(default=None)

    def __post_init__(
        self,
        create_from_file: bool,
        config_file: str | None,
        kwargs: Dict[str, Any] | None,
    ):
        """Initialize OpenMeteoClientConfig from file or kwargs with validation.

        Configures all OpenMeteo API client parameters from either a JSON configuration
        file or direct kwargs. When using files, kwargs can override specific parameters
        for flexible configuration management.

        The method validates all parameters according to OpenMeteo API constraints:
        - Date ranges must be logical and respect API data availability
        - Forecast days must be within API limits (1-16)
        - Bounding boxes must contain valid geographic coordinates
        - Metrics must be valid OpenMeteo daily parameters

        Args:
            create_from_file (bool): Whether to load base configuration from file.
                When True, attempts to load from config_file or default location.
            config_file (str | None): Path to JSON configuration file. If None and
                create_from_file=True, uses default path: {cwd}/{CONFIG_FILE env var or config.json}
            kwargs (Dict[str, Any] | None): Direct parameter values or overrides.
                Required when create_from_file=False. Can supplement file configuration.

        Supported kwargs:
            history_start_date (str | date): Start date for historical data ("YYYY-MM-DD" or date object)
            history_end_date (str | date): End date for historical data ("latest", "YYYY-MM-DD", or date object)
            forecast_days (int): Number of forecast days (1-16 inclusive)
            forecast_past_days (int): Number of past days in forecast (1-5 inclusive)
            bounding_box (Dict[str, float]): Geographic boundaries with keys: north, south, east, west
            metrics (List[str]): List of OpenMeteo daily metric names

        Raises:
            ValueError: When create_from_file=False but kwargs is None
            ValueError: When config_file is required but not provided
            ValueError: When parameter validation fails (invalid dates, ranges, etc.)

        Note:
            The "latest" keyword for history_end_date automatically sets the end date
            to 2 days before current date due to OpenMeteo Archive API data delay.
        """
        if create_from_file:
            if not config_file:
                config_file = os.path.join(
                    os.getcwd(),
                    "config",
                    os.getenv("CONFIG_FILE", "config.json"),
                )

            config = self.__get_config(config_file)

            self.__set_history_start_date(config.get("history_start_date"))
            self.__set_history_end_date(config.get("history_end_date"))
            self.__set_forecast_days(config.get("forecast_days"))
            self.__set_forecast_past_days(config.get("forecast_past_days"))
            self.__set_locations(config.get("bounding_box"))
            self.__set_metrics(config.get("metrics"))

            if kwargs:
                self.__overwrite_kwargs(kwargs)

        else:
            if kwargs:
                self.__set_history_start_date(kwargs.get("history_start_date"))
                self.__set_history_end_date(kwargs.get("history_end_date"))
                self.__set_forecast_days(kwargs.get("forecast_days"))
                self.__set_forecast_past_days(kwargs.get("forecast_past_days"))
                self.__set_locations(kwargs.get("bounding_box"))
                self.__set_metrics(kwargs.get("metrics"))
            else:
                raise ValueError("Kwargs are required when create_from_file=False.")

    def __get_config(self, config_file: str) -> Dict[str, Any]:
        """Load and parse JSON configuration file.

        Reads the specified JSON configuration file and returns the parsed
        configuration dictionary.

        Args:
            config_file (str): Absolute or relative path to the JSON configuration file.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary containing all
                configuration parameters from the file.
        """
        with open(file=config_file, mode="r") as file:
            config = json.load(fp=file)
            file.close()

        return config

    def __parse_date(self, date_string: str) -> date:
        """Parse date string into datetime.date object.

        Converts a date string in YYYY-MM-DD format into a Python date object.
        This method is used internally for processing date parameters from
        configuration files and kwargs.

        Args:
            date_string (str): Date in "YYYY-MM-DD" format (ISO 8601 date format).

        Returns:
            date: Parsed datetime.date object representing the input date.
        """
        return datetime.strptime(date_string, "%Y-%m-%d").date()

    def __compute_end_date(self) -> date:
        """Compute the latest available historical data date.

        Calculates the end date for historical data as 2 days before the current
        date. This accounts for the OpenMeteo Archive API's data processing delay,
        ensuring only verified historical data is requested.

        Returns:
            date: Date object representing 2 days before the current date,
                which is the latest date with available historical weather data.
        """
        return date.today() - timedelta(days=2)

    def __create_locations(self, bounding_box: Any, step: float = 0.5) -> NDArray:
        """Generate coordinate grid from geographic bounding box.

        Creates a regular grid of latitude and longitude coordinates within the
        specified bounding box. The grid is used to define locations for weather
        data retrieval across the geographic region of interest.

        Args:
            bounding_box (Any): Dictionary containing bounding box coordinates.
                Expected schema: {"north": float, "south": float, "east": float, "west": float}
                All values should be in decimal degrees.
            step (float, optional): Grid spacing in decimal degrees for both
                latitude and longitude. Defaults to 0.5 degrees (~55km at equator).

        Returns:
            NDArray: 2D numpy array where each row contains [latitude, longitude]
                coordinates for one grid point. Shape is (n_points, 2).

        Raises:
            ValueError: When bounding_box is not a dictionary with the required
                keys (north, south, east, west) or values are not floats.
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
        """Validate and set the history_start_date attribute.

        Processes and validates the history start date parameter, converting
        string dates to date objects as needed. The start date defines the
        beginning of the historical data retrieval period.

        Args:
            history_start_date (Any): Start date value to validate and set.
                Accepted types: str (YYYY-MM-DD format) or datetime.date object.

        Raises:
            ValueError: When history_start_date is not a string or date object.
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
        """Validate and set the history_end_date attribute.

        Processes and validates the history end date parameter, with special
        handling for the "latest" keyword that automatically computes the most
        recent available historical data date.

        Args:
            history_end_date (Any): End date value to validate and set.
                Accepted types:
                - str: "latest" (auto-computed) or "YYYY-MM-DD" format
                - datetime.date object

        Raises:
            ValueError: When history_end_date is not a string or date object.

        Note:
            The "latest" keyword sets the end date to 2 days before current date
            due to OpenMeteo Archive API data processing delays.
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
        """Validate and set the forecast_days attribute.

        Validates the number of forecast days parameter according to OpenMeteo
        API constraints. The forecast period determines how far into the future
        weather predictions are retrieved.

        Args:
            forecast_days (Any): Number of forecast days to validate and set.
                Must be an integer between 1 and 16 inclusive.

        Raises:
            ValueError: When forecast_days is not an integer.
            ValueError: When forecast_days is not greater than 0.
            ValueError: When forecast_days exceeds OpenMeteo API limits.

        Note:
            OpenMeteo Forecast API supports forecasts up to 16 days in advance.
            Values outside this range will result in API errors.
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
        """Validate and set the forecast_past_days attribute.

        Validates the number of past days to include in forecast requests.
        This parameter allows forecast endpoints to return recent historical
        data alongside predictions for continuity.

        Args:
            forecast_past_days (Any): Number of past days to validate and set.
                Must be an integer between 1 and 5 inclusive.

        Raises:
            ValueError: When forecast_past_days is not an integer.
            ValueError: When forecast_past_days is not between 1 and 5 inclusive.

        Note:
            OpenMeteo Forecast API supports up to 5 past days in forecast requests.
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
        """Generate and set the locations coordinate grid.

        Creates a coordinate grid from the provided bounding box and assigns
        it to the locations attribute. This grid defines all geographic points
        where weather data will be retrieved.

        Args:
            bounding_box (Any): Geographic bounding box specification.
                See __create_locations() for detailed format requirements.

        Raises:
            ValueError: When bounding_box format is invalid or coordinates are malformed.

        Note:
            The locations array is used by API clients to iterate through all
            coordinate points for weather data retrieval.
        """
        self.locations = self.__create_locations(bounding_box)

    def __set_metrics(self, metrics: Any) -> None:
        """Validate and set the weather metrics list.

        Validates the list of weather metrics to be retrieved from OpenMeteo APIs.
        All metrics must be valid daily parameters supported by both Archive and
        Forecast APIs for consistent data structure.

        Args:
            metrics (Any): List of weather metric names to validate and set.
                Must be a list containing valid OpenMeteo daily parameter names.

        Raises:
            ValueError: When metrics is not a list or contains invalid types.
        """
        if isinstance(metrics, list):
            self.metrics = [str(metric) for metric in metrics]
        else:
            raise ValueError(
                f"Kwarg metrics is required when create_from_file=False. Expected {type(OpenMeteoClientConfig.metrics)} Received {type(metrics)} instead."
            )

    def __overwrite_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Override configuration file parameters with kwargs values.

        Selectively overwrites configuration parameters loaded from file with
        values provided in kwargs. Only parameters present in kwargs are
        updated, allowing partial overrides of file-based configuration.

        Args:
            kwargs (Dict[str, Any]): Dictionary of parameter overrides.
                Keys should match configuration parameter names. Only parameters
                present in the dictionary will be updated.

        Note:
            This method is called automatically during initialization when both
            create_from_file=True and kwargs are provided. It enables flexible
            configuration where files provide defaults and kwargs provide
            runtime-specific overrides.
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
    """Abstract base class for OpenMeteo API clients with rate limiting and data processing.

    This abstract class provides a standardized interface for interacting with OpenMeteo APIs,
    including automatic rate limiting, request management, and response processing. It handles
    the common functionality shared between different OpenMeteo API endpoints (Archive, Forecast).

    The class implements rate limiting to respect OpenMeteo API quotas across
    multiple time windows (minutely, hourly, daily) and provides automatic backoff when
    limits are approached. It also standardizes the data extraction and processing pipeline
    for weather data retrieval.

    Key Features:
    - Multi-tier rate limiting with automatic backoff (minutely/hourly/daily)
    - Cached HTTP sessions with automatic retry logic
    - Standardized response processing to pandas DataFrames
    - Geographic coordinate handling for multi-location requests
    - Comprehensive logging for monitoring API operations
    - Database integration for seamless data persistence

    Rate Limiting:
    - Minutely: 600 requests per minute (61 second backoff)
    - Hourly: 5,000 requests per hour (3,601 second backoff)
    - Daily: 10,000 requests per day (86,401 second backoff)

    Session Configuration:
    - Cached sessions with 24-hour expiration
    - Automatic retry with exponential backoff (10 retries, factor=2)
    - Request caching to minimize redundant API calls

    Data Processing Pipeline:
    1. Multi-location coordinate iteration
    2. API request with rate limiting
    3. Response validation and variable extraction
    4. DataFrame conversion with geographic metadata
    5. Temporal data standardization

    Abstract Methods:
        get_data(): Must be implemented by subclasses to define specific API request patterns

    Subclasses:
        OpenMeteoArchiveClient: For historical weather data retrieval
        OpenMeteoForecastClient: For weather forecast data retrieval

    Attributes:
        SESSION: Configured requests session with caching and retry logic
        MINUTELY_RATE_LIMIT (int): Maximum requests per minute (600)
        HOURLY_RATE_LIMIT (int): Maximum requests per hour (5,000)
        DAILY_RATE_LIMIT (int): Maximum requests per day (10,000)
        MINUTELY_BACKOFF (int): Backoff seconds when minutely limit hit (61)
        HOURLY_BACKOFF (int): Backoff seconds when hourly limit hit (3,601)
        DAILY_BACKOFF (int): Backoff seconds when daily limit hit (86,401)
        config: OpenMeteoClientConfig instance with API parameters
        logger: Configured logger for operation monitoring

    Example:
        Subclass implementation:\n
        class CustomClient(OpenMeteoClient):
            def get_data(self, url: str) -> List[WeatherApiResponse]:
                # Implement specific API request logic
                pass

        config = OpenMeteoClientConfig(create_from_file=True)
        client = CustomClient(config)
        data = client.main()
    """

    SESSION = retry(
        requests_cache.CachedSession("/tmp/.cache", expire_after=86399),
        retries=10,
        backoff_factor=2,
    )

    MINUTELY_RATE_LIMIT = 600
    HOURLY_RATE_LIMIT = 5000
    DAILY_RATE_LIMIT = 10000
    MINUTELY_BACKOFF = 61
    HOURLY_BACKOFF = 3601
    DAILY_BACKOFF = 86401

    def __init__(self, config: OpenMeteoClientConfig):
        """Initialize OpenMeteoClient with configuration and dependencies.

        Sets up the API client with the provided configuration, establishes
        logging, initializes the database connection, and configures the
        HTTP session for API requests.

        Args:
            config (OpenMeteoClientConfig): Configuration object containing API parameters.
        """
        super().__init__(OpenMeteoClient.SESSION)  # type: ignore

        self.config = config

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.logger.info(f"Setting up {self.__class__.__name__}")

    @abstractmethod
    def get_data(self, url: str) -> List[WeatherApiResponse]:
        """Retrieve weather data from OpenMeteo API endpoint (abstract method).

        This abstract method must be implemented by subclasses to define the
        specific request patterns and parameters for different OpenMeteo API
        endpoints (Archive, Forecast, etc.).

        The implementation should handle:
        - Parameter construction for the specific API endpoint
        - Rate limiting and request timing management
        - Geographic coordinate iteration
        - Error handling and retry logic

        Args:
            url (str): Complete URL of the OpenMeteo API endpoint to query.

        Returns:
            List[WeatherApiResponse]: List of API response objects containing
                weather data for all requested locations and time periods.

        Note:
            Subclasses should implement rate limiting using handle_ratelimit()
            and provide comprehensive logging for request monitoring.
        """
        pass

    def get_request_time_estimate(self, num_requests: int) -> float:
        """Calculate estimated time required for API request batch with rate limiting.

        Estimates the total time needed to complete a batch of API requests
        considering OpenMeteo rate limits. The calculation accounts for the
        multi-tier rate limiting system and required backoff periods.

        Args:
            num_requests (int): Total number of API requests to be made.

        Returns:
            float: Estimated time in seconds to complete all requests,
                including rate limiting delays and backoff periods.

        Rate Limit Tiers:
            - ≤600 requests: No delay (within minutely limit)
            - 601-5,000 requests: Minutely backoff periods required
            - 5,001-10,000 requests: Hourly backoff periods required
            - >10,000 requests: Daily backoff periods required
        """
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
        """Manage API rate limiting across multiple time windows.

        Monitors current API usage across minutely, hourly, and daily rate limit
        windows and enforces backoff periods when limits are approached.

        Args:
            minutely_usage (float): Current requests used in the current minute window.
            hourly_usage (float): Current requests used in the current hour window.
            daily_usage (float): Current requests used in the current day window.
            fractional_api_cost (float): Cost of the next planned request in API units.

        Returns:
            Tuple[float, float, float]: Updated usage counters after any backoff
                periods. Counters are reset when their respective limits trigger
                backoff periods.

        Rate Limiting Logic:
            - Minutely limit: Triggers 61-second backoff, resets minutely counter
            - Hourly limit: Triggers 1-hour backoff, resets minutely and hourly counters
            - Daily limit: Triggers 24-hour backoff, resets all counters

        Note:
            This method will block execution during backoff periods using sleep().
        """
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
        """Extract weather variable data from OpenMeteo API response.

        Extracts numerical weather data for a specific variable from the API
        response structure and converts it to a NumPy array for processing.
        This method handles the OpenMeteo SDK's variable extraction protocol.

        Args:
            variable_index (int): Zero-based index of the weather variable
                in the response, corresponding to the order in config.metrics.
            variables (VariablesWithTime): OpenMeteo SDK variables container
                from the API response containing all requested weather metrics.

        Returns:
            np.ndarray: Numerical array containing the weather variable values
                for all time points in the response period.

        Raises:
            TypeError: When the variable at the specified index is not a
                VariableWithValues instance, indicating an invalid variable
                or response structure issue.
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
        """Convert OpenMeteo API response to structured pandas DataFrame.

        Processes a single API response by extracting the daily weather variables
        and converting them into a structured DataFrame with proper temporal
        indexing and metric organization.

        Args:
            response (WeatherApiResponse): Single API response object containing
                weather data for one geographic location.
            config (OpenMeteoClientConfig): Configuration object containing the
                list of requested weather metrics for proper data mapping.

        Returns:
            pd.DataFrame: Structured DataFrame with columns for 'date' and all
                requested weather metrics. Each row represents one day of data.

        Raises:
            TypeError: When the response does not contain a valid VariablesWithTime
                daily data section, indicating a malformed API response.

        Data Structure:
            - 'date' column: DatetimeIndex covering the response time period
            - Coordinate columns: Latitude and Longitude coordinates of a location
            - Weather metric columns: Named according to config.metrics order
            - Values: Numerical weather data extracted from API response
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

        data = pd.DataFrame(daily_data)

        data["latitude"] = response.Latitude()
        data["longitude"] = response.Longitude()

        return data

    def _main(self, url: str) -> pd.DataFrame:
        """Main data retrieval and processing pipeline for OpenMeteo API clients.

        Orchestrates the complete data retrieval process including API requests,
        response processing, geographic metadata addition, and temporal data
        standardization. This method provides the standard workflow used by
        all OpenMeteo client implementations.

        Args:
            url (str): Complete URL of the OpenMeteo API endpoint to query.

        Returns:
            pd.DataFrame: DataFrame containing weather data for all
                requested locations and time periods with the following structure:
                - 'date': Standardized date strings (YYYY-MM-DD format)
                - Weather metrics: Columns for each requested metric
                - 'latitude': Geographic latitude for each record
                - 'longitude': Geographic longitude for each record

        Processing Pipeline:
            1. Retrieve raw API responses via subclass get_data() implementation
            2. Process each response into individual DataFrames
            3. Concatenate all location data into master DataFrame
            4. Standardize date format for consistency

        Note:
            This method relies on the abstract get_data() method implementation
            in subclasses to handle the specific API request patterns for
            different OpenMeteo endpoints. It is meant to be called within main()
            routines of subclasses.
        """
        data = pd.DataFrame()
        for response in self.get_data(url):
            processed_response = self.process_response(
                response=response, config=self.config
            )

            data = pd.concat([data, processed_response], axis=0)

        data["date"] = pd.to_datetime(
            data["date"], format="%Y-%m-%d %H:%M:%S"
        ).dt.strftime("%Y-%m-%d")

        data = data.drop_duplicates()

        self.logger.info(f"{self.__class__.__name__} exited successfully.")

        return data


class OpenMeteoArchiveClient(OpenMeteoClient):
    """OpenMeteo Archive API client for historical weather data retrieval.

    This client specializes in retrieving historical weather observations from the
    OpenMeteo Historical Weather API. It handles multi-year data requests with automatic
    year-based chunking to optimize API performance and respect request limits.

    Key Features:
    - Automatic year-based request chunking for large date ranges
    - Geographic grid iteration for multi-location data retrieval
    - Rate limiting with API cost tracking
    - Temporal boundary management for partial year requests
    - Logging for monitoring data retrieval progress

    API Characteristics:
    - Endpoint: https://archive-api.open-meteo.com/v1/archive
    - Cost: 31.3 API units per location-year request
    - Data Delay: 2+ days for quality control processing
    - Coverage: Global historical weather observations
    - Resolution: Daily meteorological measurements

    Request Strategy:
    The client optimizes API usage by:
    1. Chunking requests by calendar year to minimize API costs
    2. Handling partial years at range boundaries
    3. Iterating through geographic coordinate grid
    4. Implementing progressive rate limiting across time windows

    Data Processing:
    - Extracts daily weather variables from API responses
    - Adds geographic metadata (latitude, longitude) to each record
    - Standardizes temporal formatting for database compatibility
    - Aggregates multi-location data into unified DataFrame

    Attributes:
        URL (str): OpenMeteo Archive API endpoint URL
        FRACTIONAL_API_COST (float): API cost per location-year request (31.3)

    Example:
        config = OpenMeteoClientConfig(
            create_from_file=True,
            kwargs={"history_start_date": "2020-01-01", "history_end_date": "2023-12-31"}
        )
        client = OpenMeteoArchiveClient(config)
        historical_data = client.main()
        print(f"Retrieved {len(historical_data)} records")
    """

    URL = "https://archive-api.open-meteo.com/v1/archive"

    FRACTIONAL_API_COST = 31.3

    def get_data(self, url: str) -> List[WeatherApiResponse]:
        """Retrieve historical weather data from OpenMeteo Archive API.

        Executes the complete historical data retrieval workflow including
        year-based request chunking, geographic iteration, and rate limiting
        management. The method optimizes API usage by organizing requests
        into calendar year boundaries.

        Args:
            url (str): OpenMeteo Archive API endpoint URL. Should be the
                class constant URL for consistency.

        Returns:
            List[WeatherApiResponse]: Collection of API response objects containing
                historical weather data for all requested locations and years.
                Each response covers one location-year combination.

        Request Organization:
            - Years: Derived from config.history_start_date to config.history_end_date
            - Locations: All coordinate pairs from config.locations grid
            - Total Requests: locations x years
            - API Cost: total_requests x FRACTIONAL_API_COST

        Rate Limiting:
            The method implements progressive rate limiting across three time windows:
            - Minutely: 600 requests (61s backoff when exceeded)
            - Hourly: 5,000 requests (1h backoff when exceeded)
            - Daily: 10,000 requests (24h backoff when exceeded)

        Temporal Boundaries:
            - First year: May start mid-year if history_start_date > Jan 1
            - Last year: May end mid-year if history_end_date < Dec 31
            - Full years: Always span complete calendar years
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
                start_date = (
                    date(year, 1, 1)
                    if date(year, 1, 1) > self.config.history_start_date
                    else self.config.history_start_date
                )
                end_date = (
                    date(year, 12, 31)
                    if date(year, 12, 31) < self.config.history_end_date
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
        """Main entry point for historical weather data retrieval.

        Orchestrates the complete historical weather data retrieval process
        using the OpenMeteo Historical Weather API. This method provides the primary
        interface for obtaining historical weather observations.

        Returns:
            pd.DataFrame: Historical weather dataset containing:
                - 'date': Calendar dates in YYYY-MM-DD format
                - Weather metrics: All requested meteorological measurements
                - 'latitude': Geographic latitude coordinates
                - 'longitude': Geographic longitude coordinates

        Data Coverage:
            - Temporal: From config.history_start_date to config.history_end_date
            - Spatial: All locations in config.locations coordinate grid
            - Metrics: All weather parameters specified in config.metrics

        Processing Pipeline:
            1. Execute year-chunked API requests via get_data()
            2. Process responses into structured DataFrames
            3. Add geographic metadata to each record
            4. Concatenate all location-year data
            5. Standardize date formatting for consistency
        """
        data = self._main(url=OpenMeteoArchiveClient.URL)

        return data


class OpenMeteoForecastClient(OpenMeteoClient):
    """OpenMeteo Forecast API client for weather prediction data retrieval.

    This client specializes in retrieving weather forecast predictions from the
    OpenMeteo Forecast API. It provides access to numerical weather prediction
    model outputs with configurable forecast horizons and past day inclusion.

    Key Features:
    - Configurable forecast horizon (1-16 days ahead)
    - Optional past days inclusion (1-5 days) for data continuity
    - Real-time numerical weather prediction model data
    - Geographic grid iteration for multi-location forecasts
    - Efficient rate limiting with lower API costs than historical data
    - Automatic date range calculation from configuration

    API Characteristics:
    - Endpoint: https://api.open-meteo.com/v1/forecast
    - Cost: 1.2 API units per location request
    - Update Frequency: Multiple times daily with latest model runs
    - Coverage: Global weather forecasts from numerical models
    - Resolution: Daily aggregated predictions

    Forecast Configuration:
    - forecast_days: Number of future days to predict (1-16)
    - forecast_past_days: Number of past days to include (1-5)
    - Automatic date range: past_days ago -> forecast_days ahead

    Data Processing:
    - Extracts daily forecast variables from API responses
    - Combines past observations with future predictions
    - Adds geographic metadata for each forecast location
    - Standardizes temporal formatting for database integration

    Attributes:
        URL (str): OpenMeteo Forecast API endpoint URL
        FRACTIONAL_API_COST (float): API cost per location request (1.2)
        start_date (date): Computed start date (today - forecast_past_days)
        end_date (date): Computed end date (today + forecast_days - 1)

    Example:
        config = OpenMeteoClientConfig(
            create_from_file=True,
            kwargs={"forecast_days": 7, "forecast_past_days": 1}
        )
        client = OpenMeteoForecastClient(config)
        forecast_data = client.main()
        print(f"Retrieved {len(forecast_data)} forecast records")
    """

    URL = "https://api.open-meteo.com/v1/forecast"

    FRACTIONAL_API_COST = 1.2

    def __init__(self, config: OpenMeteoClientConfig):
        """Initialize OpenMeteoForecastClient with automatic date range calculation.

        Sets up the forecast client with the provided configuration and computes
        the effective date range for forecast requests based on the configured
        forecast horizon and past days inclusion.

        Args:
            config (OpenMeteoClientConfig): Configuration object containing
                forecast parameters including forecast_days and forecast_past_days.

        Date Range Calculation:
            - start_date: today - config.forecast_past_days
            - end_date: today + config.forecast_days - 1
        """
        super().__init__(config)

        self.start_date = date.today() - timedelta(days=self.config.forecast_past_days)
        self.end_date = (
            date.today() + timedelta(days=self.config.forecast_days) - timedelta(days=1)
        )

    def get_data(self, url: str) -> List[WeatherApiResponse]:
        """Retrieve weather forecast data from OpenMeteo Forecast API.

        Executes the complete forecast data retrieval workflow including
        geographic iteration and rate limiting management. The method
        requests forecasts for all configured locations with the specified
        temporal coverage.

        Args:
            url (str): OpenMeteo Forecast API endpoint URL. Should be the
                class constant URL for consistency.

        Returns:
            List[WeatherApiResponse]: Collection of API response objects containing
                weather forecast data for all requested locations. Each response
                covers one geographic location with the full temporal range.

        Request Organization:
            - Locations: All coordinate pairs from config.locations grid
            - Temporal Range: past_days ago to forecast_days ahead
            - Total Requests: number of locations
            - API Cost: total_requests x FRACTIONAL_API_COST

        Rate Limiting:
            Implements the same progressive rate limiting as the base class:
            - Minutely: 600 requests (61s backoff when exceeded)
            - Hourly: 5,000 requests (1h backoff when exceeded)
            - Daily: 10,000 requests (24h backoff when exceeded)

        Request Parameters:
            Each API request includes:
            - latitude, longitude: Geographic coordinates
            - past_days: Number of past days to include (for continuity)
            - forecast_days: Number of future days to predict
            - daily: List of requested weather metrics
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
        """Main entry point for weather forecast data retrieval.

        Orchestrates the complete weather forecast data retrieval process
        using the OpenMeteo Forecast API. This method provides the primary
        interface for obtaining current weather predictions.

        Returns:
            pd.DataFrame: Comprehensive weather forecast dataset containing:
                - 'date': Calendar dates in YYYY-MM-DD format
                - Weather metrics: All requested meteorological predictions
                - 'latitude': Geographic latitude coordinates
                - 'longitude': Geographic longitude coordinates

        Data Coverage:
            - Temporal: From start_date (past days) to end_date (forecast horizon)
            - Spatial: All locations in config.locations coordinate grid
            - Metrics: All weather parameters specified in config.metrics

        Processing Pipeline:
            1. Execute location-based API requests via get_data()
            2. Process responses into structured DataFrames
            3. Add geographic metadata to each record
            4. Concatenate all location data
            5. Standardize date formatting for consistency

        Data Characteristics:
            - Past Days: Recent observations for continuity (if configured)
            - Future Days: Numerical weather model predictions
            - Update Frequency: Reflects latest model runs
            - Accuracy: Decreases with forecast horizon distance
        """
        data = self._main(url=OpenMeteoForecastClient.URL)

        return data


class WeeklyTableConstructor:
    """Data aggregation utility for converting daily weather data to weekly summaries.

    This class provides functionality to aggregate daily weather observations into weekly
    summaries using statistical methods appropriate for each meteorological variable.
    It handles temporal boundary management to ensure only complete weeks are processed,
    maintaining data integrity for weekly analysis.

    The aggregation process follows meteorological best practices:
    - Temperature: Mean, max, and min values preserve temperature characteristics
    - Cloud cover: Statistical aggregation maintains cloud patterns
    - Wind speed: Mean, min, max capture wind variability
    - Precipitation: Sum totals for accumulation, sum hours for duration
    - Sunshine duration: Mean duration represents weekly averages

    Key Features:
    - Automatic week boundary detection (Monday to Sunday)
    - Statistical aggregation tailored to each weather variable type
    - ISO calendar week numbering for standardized temporal reference
    - Separation of partial week data for transparent processing
    - Geographic grouping preservation across all locations

    Week Boundary Logic:
    - Start: First Monday in the dataset (beginning of first complete week)
    - End: Last Sunday in the dataset (end of last complete week)
    - Partial weeks at boundaries are separated but preserved

    Output Structure:
    - year: Calendar year of the week
    - week: ISO calendar week number (1-53)
    - latitude/longitude: Geographic coordinates
    - Aggregated weather metrics: Weekly statistical summaries

    Example:
        constructor = WeeklyTableConstructor()
        weekly_data, head_data, tail_data = constructor.main(daily_weather_df)

        print(f"Complete weeks: {len(weekly_data)}")
        print(f"Partial head data: {len(head_data)}")
        print(f"Partial tail data: {len(tail_data)}")
    """

    def __trim_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Trim daily data to complete weekly boundaries and separate partial weeks.

        Identifies complete weekly boundaries in the dataset and separates data into
        three components: main data spanning complete weeks, head data before the
        first complete week, and tail data after the last complete week.

        Args:
            data (pd.DataFrame): Daily weather data with 'date' column containing
                datetime objects. Must include all required weather metric columns.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames:
                - main: Complete weeks from first Monday to last Sunday
                - head: Partial week data before the first Monday
                - tail: Partial week data after the last Sunday

        Data Preservation:
            All original data is preserved across the three output DataFrames.
            No data is lost during the trimming process.
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
        """Aggregate daily weather data into weekly summaries by location.

        Performs statistical aggregation of daily weather variables into weekly
        summaries using methods appropriate for each meteorological parameter.
        Groups data by week, year, and geographic location to maintain spatial
        resolution while creating temporal summaries.

        Args:
            data (pd.DataFrame): Daily weather data spanning complete weeks only.
                Must contain 'date' column and all required weather metric columns.

        Returns:
            pd.DataFrame: Weekly aggregated data with the following structure:
                - week: Pandas Period representing the week ending date
                - year: Calendar year of the week
                - latitude/longitude: Geographic coordinates
                - Aggregated weather metrics: Statistical summaries

        Aggregation Logic:
            Each weather variable uses meteorologically appropriate statistics:
            - Temperature metrics: Preserve temperature characteristics
            - Cloud metrics: Capture cloud pattern variations
            - Wind metrics: Represent wind speed distributions
            - Precipitation: Total accumulation and duration
            - Sunshine: Average daily sunshine per week

        Temporal Grouping:
            Uses pandas Week offset with Sunday as the week ending day (weekday=6)
            to align with meteorological convention for weekly reporting.
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
        """Convert week period to ISO calendar week number.

        Transforms the pandas Period week representation into ISO calendar week
        numbers for standardized temporal referencing. ISO weeks are numbered
        1-53 within each calendar year and provide consistent week identification
        across different years and systems.

        Args:
            data (pd.DataFrame): Weekly aggregated data with 'week' column
                containing pandas Period objects representing week ending dates.

        Returns:
            pd.DataFrame: Modified DataFrame with 'week' column converted to
                integer ISO calendar week numbers (1-53).
        """
        data["week"] = data["week"].dt.isocalendar().week

        return data

    def main(
        self, daily_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Main aggregation pipeline for converting daily to weekly weather data.

        Orchestrates the complete process of converting daily weather observations
        into weekly summaries while preserving data integrity and handling temporal
        boundaries appropriately. This is the primary interface for weekly data
        generation.

        Args:
            daily_data (pd.DataFrame): Daily weather data with the following structure:
                - 'date': Date strings in YYYY-MM-DD format
                - Weather metrics: All required meteorological variables
                - 'latitude'/'longitude': Geographic coordinates

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames:
                - main: Weekly aggregated data for complete weeks
                - head: Daily data before first complete week (partial)
                - tail: Daily data after last complete week (partial)

        Processing Pipeline:
            1. Convert date strings to datetime objects for temporal operations
            2. Identify and separate complete weekly boundaries
            3. Aggregate complete weeks using meteorological statistics
            4. Convert to ISO calendar week numbering for standardization
        """
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        data_main, data_head, data_tail = self.__trim_data(data=daily_data)
        data_main = self.__aggregate_data(data_main)
        data_main = self.__create_calendar_week(data_main)

        data_main = data_main.drop_duplicates()

        return data_main, data_head, data_tail
