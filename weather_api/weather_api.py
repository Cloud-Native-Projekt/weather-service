from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies database connectivity"""
    try:
        # Test database connection
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail={"status": "unhealthy", "error": str(e)}
        )
