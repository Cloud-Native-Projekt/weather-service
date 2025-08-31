"""Weather Service Data Models and Database Management

This module defines the complete data model architecture, database management system,
and machine learning framework for the weather service. It provides SQLAlchemy ORM
models for weather data storage, comprehensive database operations, and Vector
Autoregression (VAR) models for weekly weather forecasting.

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

Machine Learning Framework:
- WeeklyForecastModel: VAR-based time series model for weekly weather forecasting
- Location-specific model training and prediction capabilities
- Model persistence and loading using joblib serialization

Key Features:
- Standardized weather measurement schema across all data tables
- Geographic indexing for spatial weather queries (latitude/longitude)
- Temporal indexing for efficient date and time-based operations
- Automatic bootstrap detection to prevent accidental data loss
- Data conversion utilities for pandas DataFrame to ORM object transformation
- Weekly data rollover operations for forecast-to-history transitions
- Health checks for data completeness validation and gap detection
- Multivariate time series modeling with automatic lag order selection
- Model training with stationarity preprocessing and BIC optimization
- Comprehensive logging for database and model operation monitoring

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

Machine Learning Model Architecture:
- Vector Autoregression (VAR) for multivariate time series forecasting
- First-order differencing for stationarity achievement
- Bayesian Information Criterion (BIC) for optimal lag selection
- Location-specific training for improved forecast accuracy
- Automatic undifferencing for real-scale predictions
- Configurable forecast horizons for flexible prediction timeframes

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
        location_data = db.get_data_by_location(WeeklyWeatherHistory, (40.7, -74.0))

    Maintenance operations:
        db.truncate_table(DailyWeatherForecast)
        db.rollover_weekly_data(2024, 15)

Model Operations:
    Train location-specific models:
        model = WeeklyForecastModel(location=(40.7128, -74.0060))
        model.build_model(historical_data)
        model.save('./models')

    Generate forecasts:
        loaded_model = WeeklyForecastModel.from_file('./models', location=(40.7, -74.0))
        forecast = loaded_model.forecast(horizon=4, data=historical_data)

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
- statsmodels: Vector Autoregression implementation for time series modeling
- joblib: Model serialization and persistence
- NumPy: Numerical operations and array handling
- Python logging: Operation monitoring and debugging

Note:
This module is designed to be the single source of truth for all weather data
models, database operations, and machine learning capabilities. All other services
(bootstrap, maintenance, forecast, API) should import and use these models for
data consistency and forecasting functionality.
"""

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
from warnings import deprecated

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper


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
        CheckConstraint(
            "source IN ('OpenMeteo', 'WeeklyForecastModel')",
            name="WeeklyWeatherForecast-check_source",
        ),
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
    - Health check validates completeness of historical data within configured date ranges

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
        """Check if all expected entries exist in the DailyWeatherHistory table.

        Validates data completeness by verifying that all entries within the specified
        range are present in the database table. This is useful for ensuring data
        integrity before performing operations that depend on continuous date ranges.

        Args:
            start_date (date): The beginning date of the range to check (inclusive).
            end_date (date): The ending date of the range to check (inclusive).
            table (Type[DailyWeatherHistory]: The DailyWeatherHistory table class to check.

        Returns:
            bool: True if all entries in the range exist in the table, False if any
                entries are missing.
        """
        self.__missing_entries = self.__check_missing_entries(
            start_date, end_date, table
        )

        if not self.__missing_entries:
            self.logger.info("Health check passed - no missing entries found.")

            return True
        else:
            self.logger.warning(
                f"Health check failed - {len(self.__missing_entries)} missing entries found."
            )

            return False

    @deprecated("This method is deprecated and will be removed in a future version")
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
                f"Data does not contain all expected dates. Expected start date: {start_date} Expected end date: {end_date}"
            )
            return False

    def __check_missing_entries(
        self, start_date: date, end_date: date, table: Type[DailyWeatherHistory]
    ) -> List[Tuple[date, float, float]]:
        """Check for missing date-location combinations in the DailyWeatherHistory table.

        Performs a comprehensive check to identify missing entries by cross-referencing
        all available locations with all expected dates within the specified range.
        For each location in the database, verifies that weather data exists for every
        date in the expected range and returns a list of missing date-location combinations.

        This method enables granular identification of data gaps by location and date,
        allowing for targeted data retrieval to fill specific missing entries rather
        than broad date range queries.

        Args:
            start_date (date): The beginning date of the range to check (inclusive).
            end_date (date): The ending date of the range to check (inclusive).
            table (Type[DailyWeatherHistory]): The DailyWeatherHistory table class to examine.

        Returns:
            List[Tuple[date, float, float]]: List of tuples representing missing entries,
                where each tuple contains (missing_date, latitude, longitude). Returns
                empty list if all expected date-location combinations are present in
                the database.
        """
        missing_entries = []

        available_locations = self.get_locations(table)

        expected_dates = {
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        }

        for location in available_locations:
            entries = self.DB_SESSION.scalars(
                select(table.date).where(
                    (table.latitude == float(location[0]))
                    & (table.longitude == float(location[1]))
                )
            ).all()

            if not expected_dates.issubset(entries):
                local_missing_entries = [
                    datum for datum in expected_dates if datum not in entries
                ]

                for missing_entry in local_missing_entries:
                    missing_entries.append(
                        (missing_entry, float(location[0]), float(location[1]))
                    )

        return missing_entries

    @deprecated("This method is deprecated and will be removed in a future version")
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

        self.logger.info(
            f"Database is missing the entries for the following dates: {missing_dates}"
        )

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

    @deprecated("This method is deprecated and will be removed in a future version")
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

    def get_locations(self, table: Type[WeatherTable]) -> NDArray:
        """Retrieve all unique geographic coordinate pairs from a weather table.

        Queries the database to extract all distinct latitude and longitude values
        from the specified weather table, then generates a meshgrid of all possible
        coordinate combinations.

        Args:
            table (Type[WeatherTable]): Weather table class to query. Must be a class
                that inherits from WeatherBase and contains latitude/longitude columns.

        Returns:
            NDArray: 2D NumPy array with shape (n_locations, 2) where each row contains
                [latitude, longitude] coordinates. The array contains all possible
                combinations of unique latitude and longitude values found in the table.
                Returns empty array if table contains no data.
        """
        locations = self.DB_SESSION.execute(
            select(table.latitude, table.longitude).distinct()
        ).all()

        locations = np.array(locations)

        return locations

    def get_data_by_location(
        self, table: Type[WeatherTable], location: Tuple[float, float]
    ) -> Sequence[WeatherTable]:
        """Retrieve all weather records for a specific geographic location.

        Queries the specified weather table to find all records that match the exact
        latitude and longitude coordinates provided.

        Args:
            table (Type[WeatherTable]): Weather table class to query. Must be a class
                that inherits from WeatherBase (e.g., DailyWeatherHistory,
                DailyWeatherForecast, WeeklyWeatherHistory, WeeklyWeatherForecast).
            location (Tuple[float, float]): Geographic coordinates as (latitude, longitude)
                tuple. Values should match exactly with stored coordinates in the database.

        Returns:
            Sequence[WeatherTable]: Sequence of ORM objects containing all weather
                records for the specified location. Returns empty sequence if no
                records exist for the given coordinates.
        """
        lat, lon = float(location[0]), float(location[1])

        data = self.DB_SESSION.scalars(
            select(table).where((table.latitude == lat) & (table.longitude == lon))
        ).all()

        return data

    def get_data_by_date_range(
        self,
        table: Type[DailyWeatherForecast] | Type[DailyWeatherHistory],
        start_date: date,
        end_date: date,
    ) -> Sequence[DailyWeatherHistory | DailyWeatherForecast]:
        """Retrieve weather records within a specified date range from daily tables.

        Queries the specified daily weather table to find all records with dates
        falling within the inclusive date range from start_date to end_date.

        Args:
            table (DailyWeatherForecast | DailyWeatherHistory): Daily weather table class
                to query. Must be either DailyWeatherHistory for historical data or
                DailyWeatherForecast for forecast data.
            start_date (date): Beginning date of the range to query (inclusive).
            end_date (date): Ending date of the range to query (inclusive).

        Returns:
            Sequence[DailyWeatherHistory | DailyWeatherForecast]: Sequence of ORM objects
                containing all weather records within the specified date range. Returns
                empty sequence if no records exist for the given date range.
        """
        dates = {
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        }
        data = self.DB_SESSION.scalars(select(table).where(table.date.in_(dates))).all()

        return data

    def to_dataframe(self, data: Sequence[WeatherTable]) -> pd.DataFrame:
        """Convert SQLAlchemy ORM objects to pandas DataFrame.

        Transforms a sequence of ORM objects into a pandas DataFrame for data analysis
        and manipulation. Each ORM object becomes a row in the DataFrame, with column
        names matching the ORM object attributes.

        Args:
            data (Sequence[WeatherTable]): Sequence of ORM objects to convert.
                Must be objects that inherit from WeatherBase.

        Returns:
            pd.DataFrame: DataFrame with ORM object attributes as columns and each
                object as a row. Returns empty DataFrame if data sequence is empty.
        """
        if not data:
            return pd.DataFrame()
        else:
            dataframe = pd.DataFrame(
                [
                    {
                        column.name: getattr(obj, column.name)
                        for column in obj.__table__.columns
                    }
                    for obj in data
                ]
            )

        return dataframe

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

    @property
    def get_missing_entries(self) -> List[Tuple[date, float, float]]:
        """Retrieve the list of missing date-location combinations from the last health check.

        Property that returns the cached list of missing entries identified during the most
        recent health_check() operation. This provides access to granular information about
        which specific date-location combinations are absent from the database, enabling
        targeted data retrieval for gap filling.

        The property maintains the results from the last health check to avoid repeated
        database queries when the same missing entries information is needed multiple times.
        Each tuple in the returned list represents a specific missing entry that can be
        addressed individually through targeted API requests.

        Returns:
            List[Tuple[date, float, float]]: List of tuples representing missing entries,
                where each tuple contains (missing_date, latitude, longitude). The list
                reflects the state at the time of the last health_check() call.

        Raises:
            ValueError: If health_check() has not been called yet on this WeatherDatabase
                instance, meaning no missing entries data is available to retrieve.

        Note:
            This property provides read-only access to cached health check results.
            To refresh the missing entries data, call health_check() again which will
            update the internal __missing_entries attribute.
        """
        if hasattr(self, "_WeatherDatabase__missing_entries"):
            return self.__missing_entries
        else:
            raise ValueError(
                f"{self.__class__.__name__} does not yet have a missing_entries property. Call health_check() first."
            )


WeeklyForecastModelType = TypeVar(
    "WeeklyForecastModelType", bound="WeeklyForecastModel"
)


class WeeklyForecastModel:
    """Weekly Weather Forecast Model for Machine Learning Predictions

    This class implements a Vector Autoregression (VAR) time series model for generating
    weekly weather forecasts based on historical weather patterns. The model is designed
    to be location-specific, training on historical weekly weather data for a particular
    geographic coordinate to capture local weather patterns and trends.

    Model Architecture:
    - Uses Vector Autoregression (VAR) from statsmodels for multivariate time series forecasting
    - Applies first-order differencing to achieve stationarity in the time series data
    - Automatically determines optimal lag order using Bayesian Information Criterion (BIC)
    - Supports configurable forecast horizons for flexible prediction timeframes

    Key Features:
    - Location-specific model training for improved forecast accuracy
    - Automatic lag order selection with feasibility constraints
    - Time series differencing and undifferencing for forecast generation
    - Model persistence and loading capabilities using joblib
    - Comprehensive logging for training and forecasting operations

    Training Process:
    1. Converts weekly data to datetime-indexed time series
    2. Applies first-order differencing to achieve stationarity
    3. Computes maximum feasible lag order based on data constraints
    4. Fits VAR model using ordinary least squares with BIC selection
    5. Stores trained model for future forecasting operations

    Forecasting Process:
    1. Uses trained VAR model to predict differenced values
    2. Applies cumulative sum to undifference predictions
    3. Adds back last observed values to get actual forecast levels
    4. Generates proper datetime index and metadata for forecast period

    Data Requirements:
    - Historical weekly weather data with 'year' and 'week' columns
    - Sufficient data volume for reliable model training
    - Complete time series without gaps for optimal model performance

    Attributes:
        location (Tuple[float, float]): Geographic coordinates (latitude, longitude) for the model
        model (VARResults | VARResultsWrapper | None): Trained VAR model instance
        logger: Configured logger for model operations

    Example:
        # Create and train model
        model = WeeklyForecastModel(location=(40.7128, -74.0060))
        model.build_model(historical_data)

        # Generate forecasts
        forecast = model.forecast(horizon=4, data=historical_data)

        # Save and load model
        model.save('./models')
        loaded_model = WeeklyForecastModel.from_file('./models', location=(40.7128, -74.0060))
    """

    def __init__(self, location: Tuple[float, float]) -> None:
        """Initialize WeeklyForecastModel for a specific geographic location.

        Sets up logging configuration and initializes the model instance for the specified
        geographic coordinates. The model instance starts without a trained model and
        requires build_model() to be called before forecasting.

        Args:
            location (Tuple[float, float]): Geographic coordinates as (latitude, longitude)
                tuple specifying the location for which this model will generate forecasts.
                Coordinates should match those used in the historical weather data.

        Attributes:
            location: Stores the geographic coordinates for this model instance
            model: Initialized as None; will contain the trained VAR model after build_model()
            logger: Configured logger instance for model operation monitoring
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.location = location
        self.model: VARResults | VARResultsWrapper | None = None

    def __add_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert year/week columns to datetime index for time series analysis.

        Transforms weekly data with separate 'year' and 'week' columns into a properly
        indexed time series DataFrame using ISO calendar week format. The resulting
        datetime index enables time series operations and ensures proper temporal ordering.

        Args:
            data (pd.DataFrame): Input DataFrame containing 'year' and 'week' columns
                with weekly weather data. Week numbers should follow ISO 8601 standard
                (1-53 range).

        Returns:
            pd.DataFrame: DataFrame with datetime index set to Monday of each ISO week,
                sorted by date, and proper frequency information for time series operations.

        Raises:
            ValueError: If the input DataFrame does not contain both 'year' and 'week'
                columns required for datetime index construction.

        Note:
            Uses ISO 8601 week date format (%G-W%V-%u) where weeks start on Monday.
            The resulting index represents the Monday of each calendar week.
        """
        if "year" in data.columns and "week" in data.columns:
            dt_index = pd.to_datetime(
                data["year"].astype(str) + "-W" + data["week"].astype(str) + "-1",
                format="%G-W%V-%u",
            )
            data = data.set_index(dt_index)
            data = data.sort_index()
            data.index = pd.DatetimeIndex(data=data.index, freq=None)
        else:
            raise ValueError("Data must contain 'year' and 'week' columns.")

        return data

    def __build_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series dataset for VAR model training.

        Converts raw weekly weather data into a stationary time series suitable for VAR
        modeling by applying datetime indexing, removing non-meteorological columns, and
        applying first-order differencing to achieve stationarity.

        Args:
            data (pd.DataFrame): Raw weekly weather data containing year, week, and
                meteorological measurements. Must include 'year' and 'week' columns
                for temporal identification.

        Returns:
            pd.DataFrame: Processed time series dataset with:
                - Datetime index for temporal operations
                - First-order differenced meteorological variables for stationarity
                - Metadata columns removed (year, week, idx, coordinates, source)
                - NaN values from differencing operation dropped

        Note:
            First-order differencing is applied to achieve stationarity, which is
            required for reliable VAR model estimation. The first observation is
            lost due to the differencing operation.
        """
        self.logger.info("Building time-series dataset...")

        data = self.__add_datetime_index(data)

        data = data.drop(
            columns=["year", "week", "idx", "latitude", "longitude", "source"],
            errors="ignore",
        )

        ts_data = data.diff().dropna()

        return ts_data

    def __compute_max_lags(self, data: pd.DataFrame) -> int:
        """Compute maximum feasible lag order for VAR model estimation.

        Calculates the maximum number of lags that can be used in VAR model estimation
        based on data constraints and statistical requirements. The computation ensures
        sufficient degrees of freedom for reliable parameter estimation while respecting
        practical computational limits.

        Args:
            data (pd.DataFrame): Time series dataset for which to compute maximum lags.
                Must be the differenced dataset ready for VAR modeling.

        Returns:
            int: Maximum feasible lag order that balances model complexity with
                statistical reliability. Constrained to be at least 1 and at most 156,
                with consideration for available observations and number of variables.

        Note:
            The formula ensures at least 20 observations remain for parameter estimation
            after accounting for lagged variables. The constraint (n_vars^2 + n_vars)
            reflects the number of parameters per lag in a VAR model.
        """
        n_obs = len(data)
        n_vars = len(data.columns)

        max_feasible_lags = min(
            156,
            max(1, int((n_obs - 20) / (n_vars**2 + n_vars))),
        )

        return max_feasible_lags

    def build_model(self, data: pd.DataFrame) -> None:
        """Build and train the VAR time series model using historical weather data.

        Trains a Vector Autoregression model on the provided historical weekly weather
        data. The method handles data preprocessing, optimal lag selection, and model
        fitting using ordinary least squares estimation with Bayesian Information
        Criterion for lag order selection.

        Args:
            data (pd.DataFrame): Historical weekly weather data containing year, week,
                and meteorological measurements.

        Raises:
            ValueError: If the data is insufficient for model training or lacks required
                columns ('year', 'week', meteorological variables).

        Side Effects:
            Sets self.model to the trained VARResults/VARResultsWrapper instance, which
            can then be used for generating forecasts via the forecast() method.

        Note:
            The model uses first-order differencing for stationarity and automatically
            determines the optimal lag order within feasible constraints. Training may
            take considerable time for large datasets or high-dimensional data.
        """
        self.logger.info("Building time-series model...")

        data = self.__build_dataset(data)

        model = VAR(data, freq=None)

        max_feasible_lags = self.__compute_max_lags(data)

        self.logger.info(f"Using max. lags: {max_feasible_lags}...")

        var_results = model.fit(
            maxlags=max_feasible_lags, method="ols", ic="bic", verbose=False, trend="c"
        )

        self.model = var_results

        self.logger.info("Build success!")

    def forecast(self, horizon: int, data: pd.DataFrame) -> pd.DataFrame:
        """Generate weather forecasts for the specified number of weeks ahead.

        Uses the trained VAR model to generate multi-step ahead forecasts for weekly
        weather variables. The method handles the undifferencing process to convert
        predicted changes back to actual weather values and properly formats the
        output with temporal and geographic metadata.

        Args:
            horizon (int): Number of weeks ahead to forecast. Must be a positive integer
                representing the desired forecast horizon.
            data (pd.DataFrame): Historical weekly weather data used as the basis for
                forecasting. Should be the same format as used for model training.

        Returns:
            pd.DataFrame: Forecast DataFrame containing:
                - Predicted meteorological variables for each forecast week
                - Geographic coordinates (latitude, longitude) matching the model location
                - Source identifier ('WeeklyForecastModel') for data provenance
                - Year and week columns for temporal identification
                - Reset index for consistent formatting with other weather data

        Raises:
            ValueError: If the model has not been trained (model is None) or if the
                provided data is incompatible with the trained model structure.

        Note:
            Forecasts are generated using the model's lag order and the most recent
            observations from the historical data. The undifferencing process ensures
            forecast values are in the same scale as the original observations.
        """
        self.logger.info(f"Forecasting the next {horizon} weeks...")

        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        non_differenced_values = (
            self.__add_datetime_index(data.copy())
            .drop(
                columns=["year", "week", "idx", "latitude", "longitude", "source"],
                errors="ignore",
            )
            .iloc[-1]
            .values
        )

        ts_data = self.__build_dataset(data.copy())

        p = self.model.k_ar

        forecast = self.model.forecast(y=ts_data.values[-p:], steps=horizon)

        self.logger.info("Building dataset...")

        forecast = forecast.cumsum(axis=0) + non_differenced_values

        forecast = pd.DataFrame(forecast, columns=ts_data.columns)

        last_date: date = ts_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON"
        )
        forecast = forecast.set_index(forecast_dates)

        forecast["latitude"] = self.location[0]
        forecast["longitude"] = self.location[1]
        forecast["source"] = "WeeklyForecastModel"
        forecast["year"] = pd.Series(forecast.index).dt.isocalendar().year.values
        forecast["week"] = pd.Series(forecast.index).dt.isocalendar().week.values
        forecast = forecast.reset_index(drop=True)

        return forecast

    def save(self, directory: str, file_name: str | None = None) -> str:
        """Save the trained model to disk for future use.

        Persists the complete WeeklyForecastModel instance to disk using joblib
        serialization. The saved model includes all trained parameters, metadata,
        and configuration required for loading and generating forecasts.

        Args:
            directory (str): Directory path where the model file should be saved.
                Directory must exist and be writable.
            file_name (str | None, optional): Custom filename for the saved model.
                If None, generates a standardized filename based on the model class
                name and location coordinates. Defaults to None.

        Returns:
            str: Complete file path of the saved model file, suitable for loading
                with the from_file() class method.

        Note:
            The default filename format is: 'WeeklyForecastModel_{lat}_{lon}.pkl'
            where lat and lon are the model's geographic coordinates. Existing files
            with the same name will be overwritten without warning.
        """
        if not file_name:
            file_name = (
                f"{self.__class__.__name__}_{self.location[0]}_{self.location[1]}.pkl"
            )

        path = os.path.join(directory, file_name)

        joblib.dump(self, path)

        return path

    @classmethod
    def from_file(
        cls: Type[WeeklyForecastModelType],
        directory: str,
        file_name: str | None = None,
        location: Tuple[float, float] | None = None,
    ) -> WeeklyForecastModelType:
        """Load a trained WeeklyForecastModel from disk.

        Class method that deserializes a previously saved WeeklyForecastModel instance
        from disk using joblib. The loaded model is ready for immediate use in
        generating forecasts without requiring retraining.

        Args:
            directory (str): Directory path containing the saved model file.
            file_name (str | None, optional): Specific filename of the saved model.
                If provided, takes precedence over location-based filename construction.
                Defaults to None.
            location (Tuple[float, float] | None, optional): Geographic coordinates
                used to construct the standard filename if file_name is not provided.
                Required if file_name is None. Defaults to None.

        Returns:
            WeeklyForecastModelType: Loaded WeeklyForecastModel instance with all
                trained parameters, metadata, and configuration preserved from the
                saved state.

        Raises:
            ValueError: If neither file_name nor location is provided, making it
                impossible to construct the file path for loading.
            FileNotFoundError: If the specified model file does not exist in the
                given directory.

        Note:
            Either file_name or location must be provided to locate the saved model.
            When using location, the method expects the standard filename format
            generated by the save() method.
        """
        if file_name:
            path = os.path.join(directory, file_name)
        elif location is not None:
            path = os.path.join(
                directory,
                f"{cls.__name__}_{location[0]}_{location[1]}.pkl",
            )
        else:
            raise ValueError(
                "Either file_name or location must be provided to construct path."
            )

        return joblib.load(path)
