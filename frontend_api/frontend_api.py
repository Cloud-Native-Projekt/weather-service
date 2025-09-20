"""
Weather Service Frontend API

A RESTful FastAPI application that provides access to weather data stored in a PostgreSQL database.
This service acts as the frontend API layer for retrieving historical and forecast weather data
at daily and weekly aggregation levels.

Features:
    - Health check endpoint for monitoring database connectivity
    - Location discovery for available weather data points
    - Time span queries to determine data availability ranges
    - Metrics discovery for available weather measurements
    - Daily weather data retrieval (single date, multiple dates, date ranges)
    - Weekly weather data retrieval (single week, multiple weeks, week ranges)
    - Automatic routing between historical and forecast data based on cutoff dates
    - Nearest location matching using Euclidean distance calculation

Endpoints:
    GET /health - Service and database health status
    GET /locations/{table} - Available geographic locations for a table
    GET /timespan/{table} - Available date/time ranges for a table
    GET /metrics/{table} - Available weather metrics for a table
    GET /daily/{datum}/{latitude}/{longitude}/{metrics} - Daily weather data
    GET /weekly/{calendar_week}/{latitude}/{longitude}/{metrics} - Weekly weather data

Supported Tables:
    - daily_history: Historical daily weather observations
    - daily_forecast: Future daily weather predictions
    - weekly_history: Historical weekly weather aggregations
    - weekly_forecast: Future weekly weather aggregations and additional predictions

Data Flow:
    The API automatically determines whether to query historical or forecast tables
    based on configurable cutoff dates. For requests spanning multiple time periods,
    the service combines results from both historical and forecast data sources.

Dependencies:
    - FastAPI: Web framework for building APIs
    - SQLAlchemy: Database ORM for PostgreSQL interactions
    - Pydantic: Data validation and serialization
    - NumPy: Numerical operations for location matching
    - weather_models: Custom database models and connection management

Configuration:
    - HISTORY_CUTOFF: Date boundary between historical and forecast data
    - WEEKLY_HISTORY_CUTOFF: Week boundary for weekly data routing
"""

import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Row, func, select, tuple_

from weather_models import (
    DailyWeatherForecast,
    DailyWeatherHistory,
    WeatherDatabase,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
)

DailyWeatherTable = Union[DailyWeatherHistory, DailyWeatherForecast]

WeeklyWeatherTable = Union[WeeklyWeatherForecast, WeeklyWeatherHistory]

WeatherTable = Union[
    DailyWeatherForecast,
    DailyWeatherHistory,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
]

HISTORY_CUTOFF = date.today() - timedelta(days=2)

WEEKLY_HISTORY_CUTOFF = (
    HISTORY_CUTOFF.isocalendar().year,
    HISTORY_CUTOFF.isocalendar().week,
)

TABLE_MAP: Dict[str, Type[WeatherTable]] = {
    "daily_history": DailyWeatherHistory,
    "daily_forecast": DailyWeatherForecast,
    "weekly_history": WeeklyWeatherHistory,
    "weekly_forecast": WeeklyWeatherForecast,
}

TIMESPAN_ERROR_MESSAGE = "Error retrieving available dates"
EMPTY_RESULT_SET_MESSAGE = "Request did not produce any results."

CALENDAR_WEEK_PATTERN = rf"^(?:19|20)\d{2}-(?:0[1-9]|[1-4]\d|5[0-3])$"
CALENDAR_WEEK_RANGE_PATTERN = r"^(?:19|20)\d{2}-(?:0[1-9]|[1-4]\d|5[0-3])\s*:\s*(?:19|20)\d{2}-(?:0[1-9]|[1-4]\d|5[0-3])$"
CALENDAR_WEEK_LIST_PATTERN = r"^(?:19|20)\d{2}-(?:0[1-9]|[1-4]\d|5[0-3])(?:,\s*(?:19|20)\d{2}-(?:0[1-9]|[1-4]\d|5[0-3]))*$"


class WeatherDataResponse(BaseModel):
    latitude: float
    longitude: float
    temperature_2m_mean: Optional[float]
    temperature_2m_max: Optional[float]
    temperature_2m_min: Optional[float]
    cloud_cover_mean: Optional[float]
    cloud_cover_max: Optional[float]
    cloud_cover_min: Optional[float]
    wind_speed_10m_mean: Optional[float]
    wind_speed_10m_min: Optional[float]
    wind_speed_10m_max: Optional[float]
    sunshine_duration: Optional[float]
    precipitation_sum: Optional[float]
    precipitation_hours: Optional[float]



class DailyWeatherResponse(WeatherDataResponse):
    date: date

class DailyWeatherMeansResponse(BaseModel):
    means: dict[str, float]

class WeeklyWeatherResponse(WeatherDataResponse):
    year: int
    week: int
    
class WeeklyWeatherMeansResponse(BaseModel):
    means: dict[str, float]

class LocationResponse(BaseModel):
    latitude: float
    longitude: float


class TimeSpanResponse(BaseModel):
    start: date | str
    end: date | str


class MetricsResponse(BaseModel):
    metrics: List[str]


class HealthResponse(BaseModel):
    status: str
    database: str
    message: Optional[str] = None


WeatherResponse = TypeVar("WeatherResponse", bound=WeatherDataResponse)

database: Optional[WeatherDatabase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan definition of FastAPI app.

    - Establishes database connection on startup.
    - Closes database connection on shutdown.

    Args:
        app (FastAPI): FastAPI instance.
    """
    global database
    try:
        database = WeatherDatabase()
        yield
    finally:
        if database:
            database.close()


app = FastAPI(
    title="Weather Service API",
    description="RESTful API for retrieving weather data from the weather database",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint that verifies database connectivity

    Raises:
        RuntimeError: Raised when the database has not been initialized.
        HTTPException: Raised when the database connectivity test fails.

    Returns:
        HealthResponse: HealthResponse object.
    """
    try:
        if database is None:
            raise RuntimeError("Database not initialized")

        if database.connectivity_test():
            return HealthResponse(
                status="healthy",
                database="connected",
                message="Weather database is accessible",
            )
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "database": "disconnected",
                },
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "database": "disconnected", "error": e},
        )


@app.get("/locations/{table}", response_model=List[LocationResponse])
async def get_locations(table: str) -> List[LocationResponse]:
    """Get all unique locations from the specified weather table.

    Args:
        table (str): Weather table name. Must be one of: daily_history,
                    daily_forecast, weekly_history, weekly_forecast.

    Raises:
        HTTPException: Raised when table name is invalid.
        HTTPException: Raised when the server errors out.

    Returns:
        List[LocationResponse]: List of locations. [{latitude: 0.0, longitude: 0.0},]
    """
    try:
        if table in TABLE_MAP:
            table = TABLE_MAP[table]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid table name. Must be one of: {list(TABLE_MAP.keys())}",
            )

        locations = [
            LocationResponse(latitude=float(loc[0]), longitude=float(loc[1]))
            for loc in database.get_locations(table)  # type: ignore
        ]

        return locations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving locations: {e}")


@app.get("/timespan/{table}", response_model=TimeSpanResponse)
async def get_timespan(table: str) -> TimeSpanResponse:
    """Get the available date/time range for the specified weather table.

    Args:
        table (str): Weather table name. Must be one of: daily_history,
                    daily_forecast, weekly_history, weekly_forecast.

    Raises:
        HTTPException: Status 400 when table name is invalid.
        HTTPException: Status 500 when date range retrieval fails.

    Returns:
        TimeSpanResponse: Object containing start and end dates/periods.
                         For daily tables: date format (YYYY-MM-DD).
                         For weekly tables: year-week format (YYYY-WW).
    """
    try:
        if table in TABLE_MAP:
            table = TABLE_MAP[table]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid table name. Must be one of: {list(TABLE_MAP.keys())}",
            )

        if table in (DailyWeatherHistory, DailyWeatherForecast):
            start_date = database.DB_SESSION.scalar(  # type: ignore
                select(func.min(table.date))
            )
            end_date = database.DB_SESSION.scalar(  # type: ignore
                select(func.max(table.date))
            )

            if start_date and end_date:
                timespan = TimeSpanResponse(start=start_date, end=end_date)

                return timespan
            else:
                raise HTTPException(
                    status_code=500,
                    detail=TIMESPAN_ERROR_MESSAGE,
                )
        elif table in (WeeklyWeatherHistory, WeeklyWeatherForecast):
            start_year = database.DB_SESSION.scalar(  # type: ignore
                select(func.min(table.year))
            )
            start_week = database.DB_SESSION.scalar(  # type: ignore
                select(func.min(table.year)).where(table.year == start_year)
            )
            end_year = database.DB_SESSION.scalar(  # type: ignore
                select(func.max(table.year))
            )
            end_week = database.DB_SESSION.scalar(  # type: ignore
                select(func.max(table.year)).where(table.year == start_year)
            )

            if start_year and start_week and end_year and end_week:
                start = f"{start_year}-{start_week}"
                end = f"{end_year}-{end_week}"

                timespan = TimeSpanResponse(start=start, end=end)

                return timespan
            else:
                raise HTTPException(
                    status_code=500,
                    detail=TIMESPAN_ERROR_MESSAGE,
                )
        else:
            raise HTTPException(status_code=500, detail=TIMESPAN_ERROR_MESSAGE)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{TIMESPAN_ERROR_MESSAGE}: {e}")


@app.get("/metrics/{table}", response_model=MetricsResponse)
async def get_metrics(table: str) -> MetricsResponse:
    """Get all available weather metrics for the specified table.

    Args:
        table (str): Weather table name. Must be one of: daily_history,
                    daily_forecast, weekly_history, weekly_forecast.

    Raises:
        HTTPException: Status 400 when table name is invalid.
        HTTPException: Status 500 when metrics retrieval fails.

    Returns:
        MetricsResponse: Object containing list of available metric names
                        (excludes index, date/time, and location columns).
    """
    try:
        if table in TABLE_MAP:
            table = TABLE_MAP[table]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid table name. Must be one of: {list(TABLE_MAP.keys())}",
            )

        metrics = __get_metrics(table)

        return MetricsResponse(metrics=metrics)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving available metrics: {e}"
        )


@app.get(
    "/daily/{datum}/{latitude}/{longitude}/{metrics}",
    response_model=DailyWeatherResponse | DailyWeatherMeansResponse,
)
async def get_daily_data(
    datum: str,
    latitude: float,
    longitude: float,
    metrics: str,
) -> DailyWeatherResponse | DailyWeatherMeansResponse:
    """Get daily weather data for specified date(s), location, and metrics.

    Args:
        datum (str): Date (YYYY-MM-DD), comma-separated list or range (start:end)
        latitude (float): Latitude
        longitude (float): Longitude
        metrics (str): Comma-separated metrics or 'all'

    Returns:
        DailyWeatherResponse | DailyWeatherMeansResponse: Weather data response(s)
    """
    try:
        parsed_datum = __parse_datum(datum)

        if isinstance(parsed_datum, date):
            return __process_single_date_request(
                parsed_datum, latitude, longitude, metrics
            )

        elif isinstance(parsed_datum, list):
            data = __process_date_list_request(
                parsed_datum, latitude, longitude, metrics
            )
            if isinstance(data, list) and len(data) > 1:
                means = _calculate_means(data)
                return DailyWeatherMeansResponse(means=means)
            return data[0] if data else None

        elif isinstance(parsed_datum, tuple):
            data = __process_date_range_request(
                parsed_datum, latitude, longitude, metrics
            )
            if isinstance(data, list) and len(data) > 1:
                means = _calculate_means(data)
                return DailyWeatherMeansResponse(means=means)
            return data[0] if data else None
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Request requires a date, list of dates, or a date range.\nExpected: date string, comma-separated dates, or date range\nGot {datum} instead",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {e}")


@app.get(
    "/weekly/{calendar_week}/{latitude}/{longitude}/{metrics}",
    response_model=WeeklyWeatherResponse | WeeklyWeatherMeansResponse,
)
async def get_weekly_data(
    calendar_week: str,
    latitude: float,
    longitude: float,
    metrics: str,
) -> WeeklyWeatherResponse | WeeklyWeatherMeansResponse:
    """Get weekly weather data for specified date(s), location, and metrics.

    Args:
        calendar_week (str): Calendar week in YYYY-WW format, comma-separated calendar weeks, or calendar_week_range range (start:end)
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        metrics (str): Comma-separated list of metrics or 'all'

    Raises:
        HTTPException: When parameters are invalid or data retrieval fails

    Returns:
        WeeklyWeatherResponse | WeeklyWeatherMeansResponse: Weather data response(s)
    """
    try:
        if re.fullmatch(CALENDAR_WEEK_PATTERN, calendar_week):
            return __process_single_week_request(
                calendar_week, latitude, longitude, metrics
            )

        elif re.fullmatch(CALENDAR_WEEK_LIST_PATTERN, calendar_week):
            data = __process_week_list_request(
                calendar_week, latitude, longitude, metrics
            )
            if isinstance(data, list) and len(data) > 1:
                means = _calculate_means(data)
                return WeeklyWeatherMeansResponse(means=means)
            return data[0] if data else None

        elif re.fullmatch(CALENDAR_WEEK_RANGE_PATTERN, calendar_week):
            data = __process_week_range_request(
                calendar_week, latitude, longitude, metrics
            )
            if isinstance(data, list) and len(data) > 1:
                means = _calculate_means(data)
                return WeeklyWeatherMeansResponse(means=means)
            return data[0] if data else None
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Request requires a calendar week, list of calendar weeks, or a calendar week range.\nExpected: Calendar week in YYYY-WW format, comma-separated calendar weeks, or calendar_week_range range (start:end)\nGot {calendar_week} instead",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {e}")


def _calculate_means(data_list):
    """Calculate the means of all numeric fields in a response list.

    Args:
        data_list: List of response objects (e.g. DailyWeatherResponse).

    Returns:
        Dictionary with means of all numeric fields (except coordinates and time fields).
    """
    if not data_list:
        return {}
    # Bestimme alle numerischen Felder (außer Koordinaten und Zeitfelder)
    numeric_fields = [
        k for k, v in data_list[0].__dict__.items()
        if isinstance(v, (int, float)) and k not in ("latitude", "longitude", "year", "week", "date")
    ]
    means = {}
    for field in numeric_fields:
        values = [getattr(item, field) for item in data_list if getattr(item, field) is not None]
        if values:
            means[field] = float(np.mean(values))
    return means


def __get_cols(table: Type[WeatherTable], metrics: str) -> List[Any]:
    """Get database columns for the specified table and metrics.

    Args:
        table (Type[WeatherTable]): Weather table class to query.
        metrics (str): Comma-separated metric names or "all" for all metrics.

    Returns:
        List[Any]: List of SQLAlchemy column objects including base columns
                  (date/year/week, latitude, longitude) plus requested metrics.

    Raises:
        ValueError: When table type is invalid.
    """
    if table in (DailyWeatherHistory, DailyWeatherForecast):
        base_columns = [table.date, table.latitude, table.longitude]
    elif table in (WeeklyWeatherForecast, WeeklyWeatherHistory):
        base_columns = [table.year, table.week, table.latitude, table.longitude]
    else:
        raise ValueError(
            f"Provided value for table is not of type {Type[WeatherTable]}"
        )

    available_metrics = __get_metrics(table)

    if metrics == "all":
        cols = base_columns + [getattr(table, metric) for metric in available_metrics]

        return cols
    else:
        parsed_metrics = [metric.strip() for metric in metrics.split(",")]

        selected_metrics = [
            getattr(table, metric)
            for metric in parsed_metrics
            if metric in available_metrics
        ]

        cols = base_columns + selected_metrics

        return cols


def __parse_datum(datum: str) -> date | List[date] | Tuple[date, date]:
    """Parse the datum parameter from string format.

    Args:
        datum (str): Date string in various formats:
                    - Single date: "2023-01-01"
                    - Multiple dates: "2023-01-01,2023-01-02,2023-01-03"
                    - Date range: "2023-01-01:2023-01-31"

    Returns:
        date | List[date] | Tuple[date, date]: Parsed date parameter

    Raises:
        ValueError: When date format is invalid
    """
    if ":" in datum:
        start_str, end_str = datum.split(":", 1)
        start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()

        return (start_date, end_date)

    elif "," in datum:
        date_strings = [d.strip() for d in datum.split(",")]
        dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in date_strings]

        return dates

    else:
        return datetime.strptime(datum.strip(), "%Y-%m-%d").date()


def __nearest_matching(
    latitude: float, longitude: float, table: Type[WeatherTable]
) -> Tuple[float, float]:
    """Find the nearest available location using Euclidean distance.

    Uses Euclidean distance calculation to find the closest data point
    in the specified table to the requested coordinates.

    Args:
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        table (Type[WeatherTable]): Weather table to search for locations.

    Returns:
        Tuple[float, float]: Nearest available (latitude, longitude) coordinates.

    Raises:
        HTTPException: Status 400 when coordinates are not float values.
    """
    if not isinstance(latitude, float) and not isinstance(longitude, float):
        raise HTTPException(
            status_code=400,
            detail=f"Request requires a location as latitude-longitude coordinate pairs.\nExpected: float, float\nGot {latitude} - {type(latitude)}, {longitude} - {type(longitude)} instead",
        )

    locations = database.get_locations(table)  # type: ignore

    target_location = np.array([latitude, longitude])

    distances = np.linalg.norm(locations - target_location, axis=1)
    min_index = np.argmin(distances)
    nearest_match = locations[min_index]

    return float(nearest_match[0]), float(nearest_match[1])


def __compute_cutoff_dates(
    start_date: date, end_date: date
) -> Tuple[date | None, date | None, date | None, date | None]:
    """Compute date ranges for history and forecast queries.

    Splits a date range into separate history and forecast periods based
    on HISTORY_CUTOFF to determine which tables to query.

    Args:
        start_date (date): Start date of the requested range.
        end_date (date): End date of the requested range.

    Returns:
        Tuple[date | None, date | None, date | None, date | None]:
            (history_start, history_end, forecast_start, forecast_end).
            None values indicate no data needed from that period.
    """
    history_start_date = start_date if start_date <= HISTORY_CUTOFF else None

    if history_start_date:
        history_end_date = end_date if end_date <= HISTORY_CUTOFF else HISTORY_CUTOFF
    else:
        history_end_date = None

    if start_date > HISTORY_CUTOFF:
        forecast_start_date = start_date
    elif end_date > HISTORY_CUTOFF:
        forecast_start_date = HISTORY_CUTOFF + timedelta(days=1)
    else:
        forecast_start_date = None

    forecast_end_date = end_date if forecast_start_date else None

    return (
        history_start_date,
        history_end_date,
        forecast_start_date,
        forecast_end_date,
    )


def __get_metrics(table: Type[WeatherTable]) -> List[str]:
    """Extract available weather metric column names from a table.

    Filters out non-metric columns like indexes, dates, and location data
    to return only the weather measurement column names.

    Args:
        table (Type[WeatherTable]): Weather table class to inspect.

    Returns:
        List[str]: List of weather metric column names available in the table.
    """
    return [
        metric
        for metric in table.__table__.columns.keys()
        if metric
        not in ("idx", "date", "year", "week", "latitude", "longitude", "source")
    ]


def __build_response_body(
    data: Row[Any], response_type: Type[WeatherResponse], column_names: List[str]
) -> Dict[str, Any]:
    """Build response dictionary from database row data.

    Maps database column values to response model fields, handling
    cases where requested fields may not be present in the query results.

    Args:
        data (Row[Any]): Database query result row.
        response_type (Type[WeatherResponse]): Target response model class.
        column_names (List[str]): Names of columns in the query result.

    Returns:
        Dict[str, Any]: Dictionary mapping response model fields to values,
                       with None for missing fields.
    """
    response_dict = dict(zip(column_names, data))

    body = {
        field: response_dict.get(field, None)
        for field in response_type.model_fields.keys()
    }

    return body


def __process_single_date_request(
    datum: date, latitude: float, longitude: float, metrics: str
) -> DailyWeatherResponse:
    """Process weather data request for a single date.

    Determines whether to query history or forecast table based on the
    date relative to HISTORY_CUTOFF, finds nearest location, and returns
    weather data for the specified metrics.

    Args:
        datum (date): Target date for weather data.
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        metrics (str): Comma-separated metric names or "all".

    Returns:
        DailyWeatherResponse: Weather data for the specified date and location.

    Raises:
        HTTPException: Status 500 when no data is found or query fails.
    """
    table = DailyWeatherHistory if datum <= HISTORY_CUTOFF else DailyWeatherForecast

    latitude, longitude = __nearest_matching(latitude, longitude, table)

    columns = __get_cols(table, metrics)

    response = database.DB_SESSION.execute(  # type: ignore
        select(*columns).where(
            (table.date == datum)
            & (table.latitude == latitude)
            & (table.longitude == longitude)
        )
    ).one()

    if response is not None:
        column_names = [col.key for col in columns]

        response_body = __build_response_body(
            response, DailyWeatherResponse, column_names
        )

        return DailyWeatherResponse(**response_body)
    else:
        raise HTTPException(status_code=500, detail=EMPTY_RESULT_SET_MESSAGE)


def __process_date_list_request(
    datum: List[date], latitude: float, longitude: float, metrics: str
) -> List[DailyWeatherResponse]:
    """Process weather data request for multiple specific dates.

    Splits dates into history and forecast categories based on HISTORY_CUTOFF,
    queries appropriate tables, and combines results.

    Args:
        datum (List[date]): List of target dates for weather data.
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        metrics (str): Comma-separated metric names or "all".

    Returns:
        List[DailyWeatherResponse]: Weather data for all specified dates.

    Raises:
        HTTPException: Status 400 when date format is invalid.
        HTTPException: Status 500 when no data is found or query fails.
    """
    if all(isinstance(item, date) for item in datum):
        history_dates = [item for item in datum if item <= HISTORY_CUTOFF]
        forecast_dates = [item for item in datum if item > HISTORY_CUTOFF]

        response = []

        if all(isinstance(item, date) for item in history_dates):
            latitude, longitude = __nearest_matching(
                latitude, longitude, DailyWeatherHistory
            )

            columns = __get_cols(DailyWeatherHistory, metrics)

            history_entries = database.DB_SESSION.execute(  # type: ignore
                select(*columns).where(
                    (DailyWeatherHistory.date.in_(history_dates))
                    & (DailyWeatherHistory.latitude == latitude)
                    & (DailyWeatherHistory.longitude == longitude)
                )
            ).all()

            response.extend(history_entries)

        if all(isinstance(item, date) for item in forecast_dates):
            latitude, longitude = __nearest_matching(
                latitude, longitude, DailyWeatherForecast
            )

            columns = __get_cols(DailyWeatherForecast, metrics)

            forecast_entries = database.DB_SESSION.execute(  # type: ignore
                select(*columns).where(
                    (DailyWeatherForecast.date.in_(forecast_dates))
                    & (DailyWeatherForecast.latitude == latitude)
                    & (DailyWeatherForecast.longitude == longitude)
                )
            ).all()

            response.extend(forecast_entries)

        if response:
            column_names = [col.key for col in columns]  # type: ignore

            response = [
                DailyWeatherResponse(
                    **__build_response_body(row, DailyWeatherResponse, column_names)
                )
                for row in response
            ]

            return response
        else:
            raise HTTPException(status_code=500, detail=EMPTY_RESULT_SET_MESSAGE)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Request requires a date, list of dates, or a date range.\nExpected: Date in YYYY-MM-DD format, comma-separated dates, or date range (start:end)\nGot {datum} - {type(datum)} instead",
        )


def __process_date_range_request(
    datum: Tuple[date, date], latitude: float, longitude: float, metrics: str
) -> List[DailyWeatherResponse]:
    """Process weather data request for a date range.

    Handles date ranges that may span both history and forecast periods,
    automatically splitting queries between appropriate tables.

    Args:
        datum (Tuple[date, date]): Start and end dates for the range.
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        metrics (str): Comma-separated metric names or "all".

    Returns:
        List[DailyWeatherResponse]: Weather data for all dates in the range.

    Raises:
        HTTPException: Status 400 when date format is invalid.
        HTTPException: Status 500 when no data is found or query fails.
    """
    response = []

    if all(isinstance(item, date) for item in datum):

        start_date = min(datum[0], datum[1])
        end_date = max(datum[0], datum[1])

        history_start_date, history_end_date, forecast_start_date, forecast_end_date = (
            __compute_cutoff_dates(start_date, end_date)
        )

        if history_start_date and history_end_date:
            latitude, longitude = __nearest_matching(
                latitude, longitude, DailyWeatherHistory
            )

            columns = __get_cols(DailyWeatherHistory, metrics)

            history_entries = database.DB_SESSION.execute(  # type: ignore
                select(*columns).where(
                    (DailyWeatherHistory.date >= history_start_date)
                    & (DailyWeatherHistory.date <= history_end_date)
                    & (DailyWeatherHistory.latitude == latitude)
                    & (DailyWeatherHistory.longitude == longitude)
                )
            ).all()

            response.extend(history_entries)

        if forecast_start_date and forecast_end_date:
            latitude, longitude = __nearest_matching(
                latitude, longitude, DailyWeatherForecast
            )

            columns = __get_cols(DailyWeatherForecast, metrics)

            forecast_entries = database.DB_SESSION.execute(  # type: ignore
                select(*columns).where(
                    (DailyWeatherForecast.date >= forecast_start_date)
                    & (DailyWeatherForecast.date <= forecast_end_date)
                    & (DailyWeatherForecast.latitude == latitude)
                    & (DailyWeatherForecast.longitude == longitude)
                )
            ).all()

            response.extend(forecast_entries)

        if response:
            column_names = [col.key for col in columns]  # type: ignore

            response = [
                DailyWeatherResponse(
                    **__build_response_body(row, DailyWeatherResponse, column_names)
                )
                for row in response
            ]

            return response
        else:
            raise HTTPException(status_code=500, detail=EMPTY_RESULT_SET_MESSAGE)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Request requires a date, list of dates, or a date range.\nExpected: Date in YYYY-MM-DD format, comma-separated dates, or date range (start:end)\nGot {datum} - {type(datum)} instead",
        )


def __process_single_week_request(
    calendar_week: str, latitude: float, longitude: float, metrics: str
) -> WeeklyWeatherResponse:
    """Process weather data request for a single calendar week.

    Determines whether to query history or forecast table based on the
    calendar week relative to WEEKLY_HISTORY_CUTOFF, finds nearest location, and returns
    weather data for the specified metrics.

    Args:
        calendar_week (str): Target calendar week for weather data.
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        metrics (str): Comma-separated metric names or "all".

    Returns:
        WeeklyWeatherResponse: Weather data for the specified calendar week and location.

    Raises:
        HTTPException: Status 500 when no data is found or query fails.
    """
    year, week = int(calendar_week.split("-", 1)[0]), int(
        calendar_week.split("-", 1)[1]
    )

    table = (
        WeeklyWeatherHistory
        if year <= WEEKLY_HISTORY_CUTOFF[0] and week <= WEEKLY_HISTORY_CUTOFF[1]
        else WeeklyWeatherForecast
    )

    latitude, longitude = __nearest_matching(latitude, longitude, table)

    columns = __get_cols(table, metrics)

    response = database.DB_SESSION.execute(  # type: ignore
        select(*columns).where(
            (table.year == year)
            & (table.week == week)
            & (table.latitude == latitude)
            & (table.longitude == longitude)
        )
    ).one()

    if response is not None:
        column_names = [col.key for col in columns]

        response_body = __build_response_body(
            response, WeeklyWeatherResponse, column_names
        )

        return WeeklyWeatherResponse(**response_body)
    else:
        raise HTTPException(status_code=500, detail=EMPTY_RESULT_SET_MESSAGE)


def __process_week_list_request(
    calendar_week: str, latitude: float, longitude: float, metrics: str
) -> List[WeeklyWeatherResponse]:
    """Process weather data request for multiple specific calendar weeks.

    Splits calendar weeks into history and forecast categories based on WEEKLY_HISTORY_CUTOFF,
    queries appropriate tables, and combines results.

    Args:
        calendar_week (str): Comma separated list of calendar weeks.
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        metrics (str): Comma-separated metric names or "all".

    Returns:
        List[WeeklyWeatherResponse]: Weather data for all specified calendar weeks.

    Raises:
        HTTPException: Status 400 when date format is invalid.
        HTTPException: Status 500 when no data is found or query fails.
    """
    cw_strings = [cw.strip() for cw in calendar_week.split(",")]

    calendar_weeks = [
        (int(cw.split("-")[0]), int(cw.split("-")[1])) for cw in cw_strings
    ]

    history_cws = [week for week in calendar_weeks if week <= WEEKLY_HISTORY_CUTOFF]
    forecast_cws = [week for week in calendar_weeks if week > WEEKLY_HISTORY_CUTOFF]

    response = []

    if history_cws:
        latitude, longitude = __nearest_matching(
            latitude, longitude, WeeklyWeatherHistory
        )

        columns = __get_cols(WeeklyWeatherHistory, metrics)

        history_entries = database.DB_SESSION.execute(  # type: ignore
            select(*columns).where(
                tuple_(WeeklyWeatherHistory.year, WeeklyWeatherHistory.week).in_(
                    history_cws
                )
                & (WeeklyWeatherHistory.latitude == latitude)
                & (WeeklyWeatherHistory.longitude == longitude)
            )
        ).all()

        response.extend(history_entries)

    if forecast_cws:
        latitude, longitude = __nearest_matching(
            latitude, longitude, WeeklyWeatherForecast
        )

        columns = __get_cols(WeeklyWeatherForecast, metrics)

        forecast_entries = database.DB_SESSION.execute(  # type: ignore
            select(*columns).where(
                tuple_(WeeklyWeatherForecast.year, WeeklyWeatherForecast.week).in_(
                    forecast_cws
                )
                & (WeeklyWeatherForecast.latitude == latitude)
                & (WeeklyWeatherForecast.longitude == longitude)
            )
        ).all()

        response.extend(forecast_entries)

    if response:
        column_names = [col.key for col in columns]  # type: ignore

        response = [
            WeeklyWeatherResponse(
                **__build_response_body(row, WeeklyWeatherResponse, column_names)
            )
            for row in response
        ]

        return response
    else:
        raise HTTPException(status_code=500, detail=EMPTY_RESULT_SET_MESSAGE)


def __process_week_range_request(
    calendar_week: str, latitude: float, longitude: float, metrics: str
) -> List[WeeklyWeatherResponse]:
    """Process weather data request for a calendar week range.

    Handles calendar week ranges that may span both history and forecast periods,
    automatically splitting queries between appropriate tables.

    Args:
        calendar_week (str): Start and end calendar weeks for the range.
        latitude (float): Target latitude coordinate.
        longitude (float): Target longitude coordinate.
        metrics (str): Comma-separated metric names or "all".

    Returns:
        List[WeeklyWeatherResponse]: Weather data for all calendar weeks in the range.

    Raises:
        HTTPException: Status 400 when date format is invalid.
        HTTPException: Status 500 when no data is found or query fails.
    """
    start_str, end_str = calendar_week.split(":", 1)

    start_year, start_week = start_str.split("-", 1)
    start_cw = (int(start_year), int(start_week))

    end_year, end_week = end_str.split("-", 1)
    end_cw = (int(end_year), int(end_week))

    response = []

    if start_cw <= WEEKLY_HISTORY_CUTOFF:
        latitude, longitude = __nearest_matching(
            latitude, longitude, WeeklyWeatherHistory
        )

        columns = __get_cols(WeeklyWeatherHistory, metrics)

        history_entries = database.DB_SESSION.execute(  # type: ignore
            select(*columns).where(
                (
                    tuple_(WeeklyWeatherHistory.year, WeeklyWeatherHistory.week)
                    >= start_cw
                )
                & (
                    tuple_(WeeklyWeatherHistory.year, WeeklyWeatherHistory.week)
                    <= end_cw
                )
                & (WeeklyWeatherHistory.latitude == latitude)
                & (WeeklyWeatherHistory.longitude == longitude)
            )
        ).all()

        response.extend(history_entries)

    if end_cw > WEEKLY_HISTORY_CUTOFF:
        latitude, longitude = __nearest_matching(
            latitude, longitude, WeeklyWeatherForecast
        )

        columns = __get_cols(WeeklyWeatherForecast, metrics)

        forecast_entries = database.DB_SESSION.execute(  # type: ignore
            select(*columns).where(
                (
                    tuple_(WeeklyWeatherForecast.year, WeeklyWeatherForecast.week)
                    >= start_cw
                )
                & (
                    tuple_(WeeklyWeatherForecast.year, WeeklyWeatherForecast.week)
                    <= end_cw
                )
                & (WeeklyWeatherForecast.latitude == latitude)
                & (WeeklyWeatherForecast.longitude == longitude)
            )
        ).all()

        response.extend(forecast_entries)

    if response:
        column_names = [col.key for col in columns]  # type: ignore

        response = [
            WeeklyWeatherResponse(
                **__build_response_body(row, WeeklyWeatherResponse, column_names)
            )
            for row in response
        ]

        return response
    else:
        raise HTTPException(status_code=500, detail=EMPTY_RESULT_SET_MESSAGE)
