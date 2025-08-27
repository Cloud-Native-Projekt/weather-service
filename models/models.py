import logging
import os
from abc import ABC, ABCMeta
from datetime import date, timedelta
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
)

import pandas as pd
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
    delete,
    distinct,
    func,
    inspect,
    select,
)
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import declarative_base, sessionmaker


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
        CheckConstraint("source IN ('Open Meteo', 'Placeholder')", name="check_source"),
    )


class DatabaseEngine:
    __DIALECT = "postgresql"
    __DRIVER = "psycopg2"
    __USER = os.getenv("POSTGRES_USER")
    __PASSWORD = os.getenv("POSTGRES_PASSWORD")
    __HOST = os.getenv("POSTGRES_HOST")
    __PORT = os.getenv("POSTGRES_PORT")
    __DATABASE = os.getenv("POSTGRES_DB")

    def __init__(self) -> None:
        self.__engine = create_engine(
            f"{DatabaseEngine.__DIALECT}+{DatabaseEngine.__DRIVER}://{DatabaseEngine.__USER}:{DatabaseEngine.__PASSWORD}@{DatabaseEngine.__HOST}:{DatabaseEngine.__PORT}/{DatabaseEngine.__DATABASE}",
            echo=True,
        )

    @property
    def get_engine(self):
        return self.__engine


WeatherTable = TypeVar("WeatherTable", bound=WeatherBase)


class WeatherDatabase:
    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.__engine = DatabaseEngine().get_engine

        self.DB_SESSION = sessionmaker(bind=self.__engine)()

    def create_tables(self) -> None:
        """Creates all tables from base."""
        Base.metadata.create_all(self.__engine)

    def __bootstrap(self) -> bool:
        """Checks whether the WeatherTables contain any data or should be bootstrapped.

        Returns:
            bool: True, if any WeatherTable does not contain data.
        """
        inspector = inspect(self.__engine)
        tables = inspector.get_table_names()
        expected_tables = [
            "daily_history",
            "daily_forecast",
            "weekly_history",
            "weekly_forecast",
        ]
        if tables == expected_tables:

            self.logger.info(f"Tables {expected_tables} exist in weatherdb.")

            daily_history = self.DB_SESSION.query(DailyWeatherHistory).first() is None
            daily_forecast = self.DB_SESSION.query(DailyWeatherForecast).first() is None
            weekly_history = self.DB_SESSION.query(WeeklyWeatherHistory).first() is None
            weekly_forecast = (
                self.DB_SESSION.query(WeeklyWeatherForecast).first() is None
            )

            if daily_history:
                self.logger.info("Table daily_history is empty")
            if daily_forecast:
                self.logger.info("Table daily_forecast is empty")
            if weekly_history:
                self.logger.info("Table weekly_history is empty")
            if weekly_forecast:
                self.logger.info("Table weekly_forecast is empty")

            bootstrap = (
                daily_history or daily_forecast or weekly_history
            ) or weekly_forecast

        else:
            self.logger.info(f"Tables {expected_tables} do not exist in weatherdb.")
            bootstrap = True

        return bootstrap

    def create_orm_objects(
        self, data: Dict[Any, Any] | pd.DataFrame, table: Type[WeatherTable]
    ) -> List[WeatherTable]:
        """Function to create SQLAlchemy ORM objects from Python data structures.

        Args:
            data (Dict[Any, Any] | pd.DataFrame): Data to create SQLAlchemy ORM objects from.
            table (Type[WeatherTable]): ORM class to create objects from. Can be any class that inherits from WeatherBase.

        Raises:
            TypeError: When the data does not match expected type.

        Returns:
            List[WeatherTable]: A list of ORM objects.
        """
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        elif isinstance(data, Dict):
            records = data.copy()
        else:
            raise TypeError(
                f"Data does not match expected type. Expected: {get_type_hints(self.create_orm_objects).get("data")} Got {type(data)} instead."
            )

        orm_objects = [table(**{str(k): v for k, v in row.items()}) for row in records]

        return orm_objects

    def write_data(self, orm_objects: WeatherTable | Iterable[WeatherTable]) -> None:
        """Writes SQLAlchemy ORM objects to database.

        Args:
            orm_objects (WeatherTable | Iterable[WeatherTable]: Single instance or iterable of ORM objects.

        Raises:
            TypeError: When the ORM objects do not match expected type.
        """
        if isinstance(orm_objects, WeatherBase):
            self.DB_SESSION.add(orm_objects)
        elif hasattr(orm_objects, "__iter__"):
            self.DB_SESSION.add_all(orm_objects)
        else:
            raise TypeError(
                f"Data does not match expected type. Expected: {get_type_hints(self.write_data).get("orm_objects")} Got {type(orm_objects)} instead."
            )

        self.DB_SESSION.commit()

    def health_check(
        self,
        start_date: date,
        end_date: date,
        table: Type[DailyWeatherHistory] | Type[DailyWeatherForecast],
    ) -> bool:
        date_range = self.__check_date_range(start_date, end_date, table)

        return date_range

    def __check_date_range(
        self,
        start_date: date,
        end_date: date,
        table: Type[DailyWeatherHistory] | Type[DailyWeatherForecast],
    ):
        available_dates = self.DB_SESSION.scalars(select(distinct(table.date))).all()

        expected_dates = {
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        }

        if expected_dates.issubset(available_dates):
            self.logger.info("Data exists as expected.")
            return True
        else:
            self.logger.info(
                f"Data does not contain all expected dates. Expected start date: {start_date} Expected end date: {end_date} Got: {available_dates}"
            )
            return False

    # def get_missing_dates(self, available_dates, expected_dates):
    #     missing_dates = [
    #         datum for datum in expected_dates if datum not in available_dates
    #     ]

    #     return missing_dates

    def truncate_table(self, table: Type[WeatherTable]) -> None:
        """Truncates a given WeatherTable.

        Args:
            table (Type[WeatherTable]): WeatherTable to truncate.
        """
        self.logger.info(f"Truncating table {table}...")
        self.DB_SESSION.execute(delete(table))

    def get_table(self, table: Type[WeatherTable]) -> Sequence[WeatherTable]:
        """Retrieves a given WeatherTable.

        Args:
            table (Type[WeatherTable]): WeatherTable to retrieve.

        Returns:
            Sequence[WeatherTable]: Sequence containing the data.
        """
        self.logger.info(f"Retrieving table {table}...")

        return self.DB_SESSION.scalars(select(table)).all()

    def __clone_model(
        self,
        source_obj: WeatherTable,
        target_cls: Type[WeatherTable],
        exclude: Iterable[str] = ("idx", "source"),
    ) -> WeatherTable:
        """Creates a WeatherTable object from another WeatherTable object. ORM models must be compatible with each other.

        Args:
            source_obj (WeatherTable): Source ORM object.
            target_cls (WeatherTable): Source ORM object.
            exclude (Iterable[str], optional): Defines fields to ignore. Defaults to ("idx", "source").

        Returns:
            WeatherTable: Cloned WeatherTable ORM object.
        """

        target_object = target_cls(
            **{
                k: getattr(source_obj, k)
                for k in source_obj.__table__.columns.keys()
                if k not in exclude
            }
        )

        return target_object

    def rollover_weekly_data(self, rollover_year: int, rollover_week: int) -> None:
        """Function to copy weekly data within the rollover window from WeeklyWeatherForecast to WeeklyWeatherHistory.

        Args:
            rollover_year (int): Year of data to rollover.
            rollover_week (int): Calendar week of data to rollover.
        """
        source_data = self.DB_SESSION.scalars(
            select(WeeklyWeatherForecast).where(
                (WeeklyWeatherForecast.year == rollover_year)
                and (WeeklyWeatherForecast.week == rollover_week)
            )
        ).all()

        self.DB_SESSION.delete(source_data)

        target_data = [
            self.__clone_model(source_obj=entry, target_cls=WeeklyWeatherHistory)
            for entry in source_data
        ]

        self.write_data(target_data)

    def close(self) -> None:
        """Closes the session to the database. All operations should be completed before calling this method."""
        self.logger.info("Closing Database Session...")
        self.DB_SESSION.close()

    @property
    def bootstrap(self) -> bool:
        """Get bootstrap property of WeatherDatabase instance.

        Returns:
            bool: If True, continue with bootstrapping.
        """
        return self.__bootstrap()
