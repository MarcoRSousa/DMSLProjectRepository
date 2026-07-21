
from fastapi import FastAPI

from database.database import engine
from database.database import wait_for_database
from database.models import Base


wait_for_database()
Base.metadata.create_all(bind=engine)

app = FastAPI()


@app.get("/")
def root():

    return {"message": "Backend running"}
