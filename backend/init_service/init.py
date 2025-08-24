from openmeteo_client import OpenMeteoArchiveClient, OpenMeteoForecastClient

from models import Base, DatabaseEngine

if __name__ == "__main__":
    engine = DatabaseEngine().get_engine
    Base.metadata.create_all(engine)
    ArchiveClient = OpenMeteoArchiveClient()
    ArchiveClient.main()
    ForecastClient = OpenMeteoForecastClient()
    ForecastClient.main()
