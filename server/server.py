from fastapi import FastAPI, Depends, FastAPI, HTTPException, status, Cookie, Request
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    #lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.mount("/", StaticFiles(directory="../client/build", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app")