import logging
import os
from abc import ABC, ABCMeta
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Type, TypeVar, get_type_hints
from warnings import deprecated

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
    """PostgreSQL Database Engine Configuration and Connection Management

    The class automatically constructs database connection URLs using the PostgreSQL
    dialect with the psycopg2 driver, and retrieves all connection parameters from
    environment variables for security and configuration management.

    Environment Variables Required:
    - POSTGRES_USER: Database username for authentication
    - POSTGRES_PASSWORD: Database password for authentication
    - POSTGRES_HOST: Database server hostname or IP address
    - POSTGRES_PORT: Database server port number
    - POSTGRES_DB: Target database name

    Attributes:
        __DIALECT (str): Database dialect identifier ("postgresql")
        __DRIVER (str): Database driver identifier ("psycopg2")
        __USER (str): Database username from environment
        __PASSWORD (str): Database password from environment
        __HOST (str): Database host from environment
        __PORT (str): Database port from environment
        __DATABASE (str): Database name from environment
    """

    __DIALECT = "postgresql"
    __DRIVER = "psycopg2"
    __USER = os.getenv("POSTGRES_USER")
    __PASSWORD = os.getenv("POSTGRES_PASSWORD")
    __HOST = os.getenv("POSTGRES_HOST")
    __PORT = os.getenv("POSTGRES_PORT")
    __DATABASE = os.getenv("POSTGRES_DB")

    def __init__(self) -> None:
        """Initialize DatabaseEngine with PostgreSQL connection."""
        self.__engine = create_engine(
            f"{DatabaseEngine.__DIALECT}+{DatabaseEngine.__DRIVER}://{DatabaseEngine.__USER}:{DatabaseEngine.__PASSWORD}@{DatabaseEngine.__HOST}:{DatabaseEngine.__PORT}/{DatabaseEngine.__DATABASE}",
            echo=False,
        )

    @property
    def get_engine(self):
        """SQLAlchemy database engine.

        Returns:
            sqlalchemy.engine.Engine: Configured SQLAlchemy engine instance
                ready for database operations.
        """
        return self.__engine


WeatherTable = TypeVar("WeatherTable", bound=WeatherBase)


class WeatherDatabase:
    """Weather Database Management Class

    This class provides an interface for managing weather data persistence
    in a PostgreSQL database. It handles database connections, table operations, data
    manipulation, and maintenance routines for the weather service.

    The class manages four primary data tables:
    - DailyWeatherHistory: Historical daily weather observations
    - DailyWeatherForecast: Current daily weather forecasts
    - WeeklyWeatherHistory: Aggregated weekly historical data
    - WeeklyWeatherForecast: Aggregated weekly forecast data

    Key Features:
    - Automatic bootstrap detection to prevent accidental data loss
    - ORM object creation from pandas DataFrames and dictionaries
    - Data rollover operations for weekly forecast-to-history transitions
    - Table truncation and retrieval operations
    - Session management with proper cleanup

    The class uses SQLAlchemy ORM for database operations and includes
    logging for monitoring database activities.

    Attributes:
        logger: Configured logger instance for database operations
        DB_SESSION: SQLAlchemy session for database transactions
        bootstrap: Property indicating if database initialization is needed

    Example:
        db = WeatherDatabase()
        if db.bootstrap:
            db.create_tables()
            # Populate initial data
        db.close()
    """

    def __init__(self) -> None:
        """Initialize WeatherDatabase instance.

        Sets up logging configuration, establishes database engine connection,
        and creates a new database session for operations.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.__engine = DatabaseEngine().get_engine

        self.DB_SESSION = sessionmaker(bind=self.__engine)()

    def create_tables(self) -> None:
        """Create all weather data tables in the database.

        Uses SQLAlchemy metadata to create all tables defined in the Base
        declarative registry. This method is typically called during bootstrap
        operations to ensure all required tables exist.
        """
        Base.metadata.create_all(self.__engine)

    def __bootstrap(self) -> bool:
        """Check whether the database requires bootstrap initialization.

        Examines the database to determine if all expected tables exist and contain
        data. Bootstrap is required if any tables are missing or empty.

        Returns:
            bool: True if any WeatherTable does not exist or contains no data,
                 indicating bootstrap is needed. False if all tables exist and
                 contain data.
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
        """Create SQLAlchemy ORM objects from data structures.

        Converts pandas DataFrames or dictionaries into SQLAlchemy ORM objects for database persistence.
        Handles data type validation and conversion for compatibility with the specified table schema.

        Args:
            data (Dict[Any, Any] | pd.DataFrame): Source data to convert. Can be
                a pandas DataFrame with records as rows, or a dictionary with
                column names as keys.
            table (Type[WeatherTable]): Target ORM class of type WeatherTable. Determines the table schema and structure for
                the created objects.

        Raises:
            TypeError: When the data parameter is not a supported type
                (DataFrame or Dict).

        Returns:
            List[WeatherTable]: List of instantiated ORM objects
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
        """Write SQLAlchemy ORM objects to the database.

        Persists ORM objects to the database using the current session.
        Supports both single objects and collections of objects. All operations
        are committed as a single transaction.

        Args:
            orm_objects (WeatherTable | Iterable[WeatherTable]): Single ORM object
                or iterable collection of ORM objects to persist.

        Raises:
            TypeError: When orm_objects is not a WeatherBase instance or iterable
                collection of WeatherBase instances.

        Note:
            This method commits the transaction immediately. Ensure all objects
            are properly validated before calling this method.
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

    @deprecated(
        "This method is deprecated in it's current state and will be changed in a future version"
    )
    def health_check(
        self,
        start_date: date,
        end_date: date,
        table: Type[DailyWeatherHistory] | Type[DailyWeatherForecast],
    ) -> bool:
        date_range = self.__check_date_range(start_date, end_date, table)

        return date_range

    @deprecated(
        "This method is deprecated in it's current state and will be changed in a future version"
    )
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

    @deprecated(
        "This method is deprecated in it's current state and will be changed in a future version"
    )
    def get_missing_dates(self, available_dates, expected_dates):
        missing_dates = [
            datum for datum in expected_dates if datum not in available_dates
        ]

        return missing_dates

    def truncate_table(self, table: Type[WeatherTable]) -> None:
        """Truncates a specified weather table.

        Executes a DELETE operation to remove all records from the specified
        table while preserving the table structure.

        Args:
            table (Type[WeatherTable]): Weather table class to truncate. Must be
                a class that inherits from WeatherBase.

        Warning:
            This operation permanently removes all data from the specified table.
        """
        self.logger.info(f"Truncating table {table}...")
        self.DB_SESSION.execute(delete(table))

    def get_table(self, table: Type[WeatherTable]) -> Sequence[WeatherTable]:
        """Retrieve all records from a specified weather table.

        Executes a SELECT operation to fetch all records from the specified
        table and returns them as ORM objects.

        Args:
            table (Type[WeatherTable]): Weather table class to retrieve.

        Returns:
            Sequence[WeatherTable]: Sequence containing all records from the table
                as ORM objects. Returns empty sequence if table is empty.
        """
        self.logger.info(f"Retrieving table {table}...")

        return self.DB_SESSION.scalars(select(table)).all()

    def __clone_model(
        self,
        source_obj: WeatherTable,
        target_cls: Type[WeatherTable],
        exclude: Iterable[str] = ("idx", "source"),
    ) -> WeatherTable:
        """Create a new ORM object by cloning data from an existing object.

        Copies all compatible attributes from a source ORM object to create a new
        instance of a different (but compatible) table class. This is used for
        data migration between related tables, such as moving forecast data to
        historical tables.

        Args:
            source_obj (WeatherTable): Source ORM object to copy data from.
            target_cls (Type[WeatherTable]): Target ORM class to create. Must have
                compatible schema with the source object.
            exclude (Iterable[str], optional): Column names to exclude from copying.
                Defaults to ("idx", "source") to avoid primary key conflicts and
                source-specific fields.

        Returns:
            WeatherTable: New ORM object instance of target_cls with copied data
                from source_obj.
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
        """Transfer weekly forecast data to historical data for a completed week.

        Moves weekly forecast records for the specified year and week from the
        WeeklyWeatherForecast table to the WeeklyWeatherHistory table. This
        operation is performed when a forecast period has passed and becomes
        historical data.

        Args:
            rollover_year (int): Year of the data to rollover (e.g., 2024).
            rollover_week (int): ISO calendar week number to rollover (1-53).

        Note:
            This operation performs both deletion from the forecast table and
            insertion into the history table as part of a single transaction.
            The rollover preserves all weather data while changing its classification
            from forecast to historical.
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
        """Close the database session and release connections.

        Properly terminates the database session to prevent connection leaks.
        This method should be called when database operations are complete.
        """
        self.logger.info("Closing Database Session...")
        self.DB_SESSION.close()

    @property
    def bootstrap(self) -> bool:
        """Determine if database bootstrap initialization is required.

        Property that checks the current state of the database to determine
        whether bootstrap operations should be performed. Bootstrap is needed
        when tables are missing or empty.

        Returns:
            bool: True if bootstrap operations are needed (tables missing or empty),
                False if database is properly initialized with data.
        """
        return self.__bootstrap()
