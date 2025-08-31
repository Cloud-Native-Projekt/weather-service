from contextlib import asynccontextmanager
from datetime import date
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from weather_models import (
    DailyWeatherForecast,
    DailyWeatherHistory,
    WeatherDatabase,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
)


class WeatherDataResponse(BaseModel):
    idx: int
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


class WeeklyWeatherResponse(WeatherDataResponse):
    year: int
    week: int


class WeeklyForecastResponse(WeeklyWeatherResponse):
    source: str


class LocationResponse(BaseModel):
    latitude: float
    longitude: float


class TimeSpanResponse(BaseModel):
    start: date | str
    end: date | str


class HealthResponse(BaseModel):
    status: str
    database: str
    message: Optional[str] = None


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


def start():
    """_summary_"""
    uvicorn.run("weather_api.weather_api:app", host="0.0.0.0", port=8000, reload=True)


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
            detail={"status": "unhealthy", "database": "disconnected", "error": str(e)},
        )


@app.get("/locations/{table}", response_model=List[LocationResponse])
async def get_locations(table: str) -> List[LocationResponse]:
    """Get all unique locations from the specified weather table.

    Args:
        table (str): WeatherTable to retrieve locations from.

    Raises:
        HTTPException: Raised when table name is invalid.
        HTTPException: Raised when the server errors out.

    Returns:
        List[LocationResponse]: List of locations. [(Latitude, Longitude)]
    """
    try:
        table_map = {
            "daily_history": DailyWeatherHistory,
            "daily_forecast": DailyWeatherForecast,
            "weekly_history": WeeklyWeatherHistory,
            "weekly_forecast": WeeklyWeatherForecast,
        }

        if table not in table_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid table name. Must be one of: {list(table_map.keys())}",
            )

        locations = [
            LocationResponse(latitude=float(loc[0]), longitude=float(loc[1]))
            for loc in database.get_locations(table_map[table])  # type: ignore
        ]

        return locations

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving locations: {str(e)}"
        )


@app.get("/timespan/{table}", response_model=TimeSpanResponse)
async def get_timespan(table: str) -> TimeSpanResponse:
    """Get all available dates from daily weather tables"""
    try:
        table_map = {
            "daily_history": DailyWeatherHistory,
            "daily_forecast": DailyWeatherForecast,
            "weekly_history": WeeklyWeatherHistory,
            "weekly_forecast": WeeklyWeatherForecast,
        }

        if table in table_map:
            table = table_map[table]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid table name. Must be one of: {list(table_map.keys())}",
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
                    detail="Error retrieving available dates",
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
                    detail="Error retrieving available dates",
                )
        else:
            raise HTTPException(
                status_code=500, detail="Error retrieving available dates"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving available dates: {str(e)}"
        )
