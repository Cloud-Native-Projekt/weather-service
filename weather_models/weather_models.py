"""Weather Service Data Models and Database Management

This module defines the complete data model architecture and database management
system for the weather service. It provides SQLAlchemy ORM models for weather data
storage and a comprehensive database interface for weather data operations.

Core Components:

Database Models:
- WeatherBase: Abstract base class defining common weather measurement schema
- DailyWeatherHistory: Historical daily weather observations storage
- DailyWeatherForecast: Daily weather forecast predictions storage
- WeeklyWeatherHistory: Aggregated weekly historical weather data
- WeeklyWeatherForecast: Aggregated weekly forecast predictions

Database Management:
- DatabaseEngine: PostgreSQL connection and engine configuration
- WeatherDatabase: High-level interface for all database operations

Key Features:
- Standardized weather measurement schema across all data tables
- Geographic indexing for spatial weather queries (latitude/longitude)
- Temporal indexing for efficient date and time-based operations
- Automatic bootstrap detection to prevent accidental data loss
- Data conversion utilities for pandas DataFrame to ORM object transformation
- Weekly data rollover operations for forecast-to-history transitions
- Comprehensive logging for database operation monitoring

Data Schema:
All weather tables inherit a common set of meteorological measurements:
- Temperature metrics (mean, max, min) in Celsius at 2m height
- Cloud cover percentages (mean, max, min)
- Wind speed measurements (mean, max, min) in m/s at 10m height
- Precipitation data (total mm, duration hours)
- Sunshine duration in seconds
- Geographic coordinates (latitude, longitude)

Database Configuration:
The system uses PostgreSQL with psycopg2 adapter and requires the following
environment variables for connection:
- POSTGRES_USER: Database username
- POSTGRES_PASSWORD: Database password
- POSTGRES_HOST: Database server hostname
- POSTGRES_PORT: Database server port
- POSTGRES_DB: Target database name

Usage Patterns:

Bootstrap Operations:
    db = WeatherDatabase()
    if db.bootstrap:
        db.create_tables()
        # Populate initial data via bootstrap service

Data Operations:
    Convert DataFrame to ORM objects:
        objects = db.create_orm_objects(weather_df, DailyWeatherHistory)
        db.write_data(objects)

    Query data:
        history = db.get_table(DailyWeatherHistory)

    Maintenance operations:
        db.truncate_table(DailyWeatherForecast)
        db.rollover_weekly_data(2024, 15)

Session Management:
    try:
        db = WeatherDatabase()
        # Perform operations
    finally:
        db.close()

Dependencies:
- SQLAlchemy: ORM and database abstraction layer
- pandas: Data manipulation and DataFrame operations
- PostgreSQL: Primary database backend with psycopg2 driver
- Python logging: Operation monitoring and debugging

Note:
This module is designed to be the single source of truth for all weather data
models and database operations. All other services (bootstrap, maintenance, API)
should import and use these models for data consistency.
"""

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
    UniqueConstraint,
    create_engine,
    delete,
    distinct,
    inspect,
    select,
)
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import declarative_base, sessionmaker


class CombinedMeta(DeclarativeMeta, ABCMeta):
    """Combined metaclass for SQLAlchemy declarative base with abstract base class support."""

    pass


Base = declarative_base(metaclass=CombinedMeta)


class WeatherBase(Base, ABC):
    """Abstract base class for all weather data tables.

    This abstract class defines the common schema and structure shared by all weather
    data tables in the system. It provides standardized meteorological measurements
    and geographic positioning that are consistent across daily and weekly, historical
    and forecast data tables.

    All concrete weather tables inherit from this base class and add their own
    temporal identification columns (date, year/week) as appropriate.
    """

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
    """Historical daily weather observations data table.

    Table Structure:
    - Inherits all weather measurements from WeatherBase
    - Adds date column for temporal identification
    - Geographic indexing on latitude/longitude for spatial queries
    - Date indexing for efficient temporal range queries
    """

    __tablename__ = "daily_history"

    date = Column(Date, index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "date",
            "latitude",
            "longitude",
            name="DailyWeatherHistory-entry-unique-constraint",
        ),
    )


class DailyWeatherForecast(WeatherBase):
    """Daily weather forecast predictions data table.

    Table Structure:
    - Inherits all weather measurements from WeatherBase
    - Adds date column for forecast validity date
    - Geographic indexing for spatial forecast queries
    - Date indexing for temporal forecast range selection
    """

    __tablename__ = "daily_forecast"

    date = Column(Date, index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "date",
            "latitude",
            "longitude",
            name="DailyWeatherForecast-entry-unique-constraint",
        ),
    )


class WeeklyWeatherHistory(WeatherBase):
    """Historical weekly aggregated weather data table.

    Table Structure:
    - Inherits all weather measurements from WeatherBase (as weekly aggregates)
    - Uses year and week columns for temporal identification
    - Geographic indexing for spatial analysis across regions
    - Composite indexing on year/week for efficient temporal queries
    """

    __tablename__ = "weekly_history"

    year = Column(Integer, index=True, nullable=False)
    week = Column(Integer, index=True, nullable=False)


class WeeklyWeatherForecast(WeatherBase):
    """Weekly aggregated weather forecast predictions data table.

    Table Structure:
    - Inherits all weather measurements from WeatherBase (as weekly aggregates)
    - Uses year and week columns for forecast validity period
    - Source column tracks the origin of forecast data
    - Check constraint ensures valid source values
    """

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
        try:
            self.DB_SESSION.commit()
        except Exception as e:
            self.logger.error(f"Error during writing data: {e}")
            self.logger.info("Rolling back transaction...")
            self.DB_SESSION.rollback()

    def health_check(
        self,
        start_date: date,
        end_date: date,
        table: Type[DailyWeatherHistory],
    ) -> bool:
        """Check if all expected dates exist in the specified daily weather table.

        Validates data completeness by verifying that all dates within the specified
        range are present in the database table. This is useful for ensuring data
        integrity before performing operations that depend on continuous date ranges.

        Args:
            start_date (date): The beginning date of the range to check (inclusive).
            end_date (date): The ending date of the range to check (inclusive).
            table (Type[DailyWeatherHistory]: The DailyWeatherHistory table class to check.

        Returns:
            bool: True if all dates in the range exist in the table, False if any
                dates are missing.
        """
        date_range = self.__check_date_range(start_date, end_date, table)

        return date_range

    def __check_date_range(
        self,
        start_date: date,
        end_date: date,
        table: Type[DailyWeatherHistory],
    ) -> bool:
        """Internal method to verify date range completeness in DailyWeatherHistory.

        Compares the expected date range against the actual dates present in the
        database table. Logs the results of the comparison for monitoring and
        debugging purposes.

        Args:
            start_date (date): The beginning date of the expected range (inclusive).
            end_date (date): The ending date of the expected range (inclusive).
            table (Type[DailyWeatherHistory]: The DailyWeatherHistory class to examine.

        Returns:
            bool: True if all expected dates are present in the table, False if any
                dates are missing. Also logs informational messages about the
                comparison results.
        """
        available_dates = self.get_date_range(table)

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

    def get_missing_dates(
        self,
        start_date: date,
        end_date: date,
        table: Type[DailyWeatherHistory],
    ):
        """Identify missing dates within a specified range in DailyWeatherHistory.

        Compares the expected date range against the actual dates present in the
        database and returns a list of dates that are missing. This is useful for
        identifying data gaps that need to be filled.

        Args:
            start_date (date): The beginning date of the range to check (inclusive).
            end_date (date): The ending date of the range to check (inclusive).
            table (Type[DailyWeatherHistory]: The DailyWeatherHistory class to examine.

        Returns:
            List[date]: List of date objects representing the missing dates within
                the specified range. Returns empty list if all dates are present.
        """
        available_dates = self.get_date_range(table)

        expected_dates = {
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        }

        missing_dates = [
            datum for datum in expected_dates if datum not in available_dates
        ]

        return missing_dates

    def get_date_range(
        self, table: Type[DailyWeatherHistory] | Type[DailyWeatherForecast]
    ) -> Sequence[date]:
        """Retrieve all unique dates present in a daily weather table.

        Queries the database to get a distinct list of all dates that have weather
        data in the specified table. This is useful for understanding the temporal
        coverage of the dataset and identifying available data periods.

        Args:
            table (Type[DailyWeatherHistory] | Type[DailyWeatherForecast]): The daily
                weather table class to query. Must be either DailyWeatherHistory or
                DailyWeatherForecast.

        Returns:
            Sequence[date]: Sequence of unique date objects representing all dates
                that have weather data in the table. Returns empty sequence if the
                table contains no data.
        """
        return self.DB_SESSION.scalars(select(distinct(table.date))).all()

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
