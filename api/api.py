import datetime
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import table, text
from sqlmodel import Field, Session, SQLModel, create_engine, select


class WeatherTableBase(SQLModel):
    idx: int | None = Field(default=None, primary_key=True)
    latitude: float = Field(index=True)
    longitude: float = Field(index=True)
    date: datetime.date = Field(index=True)
    temperature_2m_mean: float | None = Field(default=None)
    temperature_2m_max: float | None = Field(default=None)
    temperature_2m_min: float | None = Field(default=None)
    cloud_cover_mean: float | None = Field(default=None)
    cloud_cover_max: float | None = Field(default=None)
    cloud_cover_min: float | None = Field(default=None)
    wind_gusts_10m_mean: float | None = Field(default=None)
    wind_speed_10m_mean: float | None = Field(default=None)
    wind_gusts_10m_min: float | None = Field(default=None)
    wind_speed_10m_min: float | None = Field(default=None)
    wind_speed_10m_max: float | None = Field(default=None)
    wind_gusts_10m_max: float | None = Field(default=None)
    sunshine_duration: float | None = Field(default=None)
    precipitation_sum: float | None = Field(default=None)
    precipitation_hours: float | None = Field(default=None)


# class HistoricWeather(WeatherTableBase, table=True):
#     pass


# class CurrentWeather(WeatherTableBase, table=True):
#     pass


class DBConnection:
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_PORT = os.getenv("MYSQL_PORT")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

    def __str__(self) -> str:
        return f"mysql+pymysql://{DBConnection.MYSQL_USER}:{DBConnection.MYSQL_PASSWORD}@{DBConnection.MYSQL_HOST}:{DBConnection.MYSQL_PORT}/{DBConnection.MYSQL_DATABASE}"


# Create engine
engine = create_engine(DBConnection().__str__(), echo=True)


# def create_db_and_tables():
#     SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies database connectivity"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail={"status": "unhealthy", "error": str(e)}
        )


@app.post("/history/post")
async def write_history(session: Session = Depends(get_session)):
    pass
