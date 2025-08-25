from openmeteo_client import OpenMeteoArchiveClient, OpenMeteoForecastClient

from models import Base, DatabaseEngine

if __name__ == "__main__":
    engine = DatabaseEngine().get_engine
    Base.metadata.create_all(engine)
    ArchiveClient = OpenMeteoArchiveClient(create_from_file=True)
    ArchiveClient.main()
    ForecastClient = OpenMeteoForecastClient(create_from_file=True)
    ForecastClient.main()
