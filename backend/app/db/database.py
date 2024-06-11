# from sqlalchemy import create_engine, Column, Integer, Float, String
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from app.core.config import settings

# engine = create_engine(settings.database_url)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class AirQualityData(Base):
#     __tablename__ = 'air_quality_data'
#     id = Column(Integer, primary_key=True, index=True)
#     pm10 = Column(Float)
#     prediction = Column(Float)
#     timestamp = Column(String)

# def init_db():
#     Base.metadata.create_all(bind=engine)

# init_db()