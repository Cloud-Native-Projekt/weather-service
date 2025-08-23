import os
from abc import ABC, ABCMeta

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import declarative_base


class CombinedMeta(DeclarativeMeta, ABCMeta):
    pass


Base = declarative_base(metaclass=CombinedMeta)


class WeatherBase(Base, ABC):
    __abstract__ = True

    idx = Column(Integer, primary_key=True, autoincrement=True)
    latitude = Column(Float, index=True, nullable=False)
    longitude = Column(Float, index=True, nullable=False)
    temperature_2m_mean = Column(Float)
    temperature_2m_max = Column(Float)
    temperature_2m_min = Column(Float)
    cloud_cover_mean = Column(Float)
    cloud_cover_max = Column(Float)
    cloud_cover_min = Column(Float)
    wind_speed_10m_mean = Column(Float)
    wind_speed_10m_min = Column(Float)
    wind_speed_10m_max = Column(Float)
    sunshine_duration = Column(Float)
    precipitation_sum = Column(Float)
    precipitation_hours = Column(Float)


class DailyWeatherHistory(WeatherBase):
    __tablename__ = "daily_history"

    date = Column(Date, index=True, nullable=False)


class DailyWeatherForecast(WeatherBase):
    __tablename__ = "daily_forecast"

    date = Column(Date, index=True, nullable=False)


class WeeklyWeatherHistory(WeatherBase):
    __tablename__ = "weekly_history"

    year = Column(Integer, index=True, nullable=False)
    week = Column(Integer, index=True, nullable=False)


class WeeklyWeatherForecast(WeatherBase):
    __tablename__ = "weekly_forecast"

    year = Column(Integer, index=True, nullable=False)
    week = Column(Integer, index=True, nullable=False)
    source = Column(String(length=32), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "source IN ('Open Meteo', 'Placeholder')", name="check_source"
        ),  # TODO: replace placeholder
    )


class DatabaseEngine:
    __DIALECT = "mysql"
    __DRIVER = "pymysql"
    __USER = os.getenv("MYSQL_USER")
    __PASSWORD = os.getenv("MYSQL_PASSWORD")
    __HOST = os.getenv("MYSQL_HOST")
    __PORT = os.getenv("MYSQL_PORT")
    __DATABASE = os.getenv("MYSQL_DATABASE")

    def __init__(self) -> None:
        self.__engine = create_engine(
            f"{DatabaseEngine.__DIALECT}+{DatabaseEngine.__DRIVER}://{DatabaseEngine.__USER}:{DatabaseEngine.__PASSWORD}@{DatabaseEngine.__HOST}:{DatabaseEngine.__PORT}/{DatabaseEngine.__DATABASE}",
            echo=True,
        )

    @property
    def get_engine(self):
        return self.__engine
