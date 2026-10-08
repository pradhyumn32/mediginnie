# Medijinn Main Application Entrypoint
# Runs the FastAPI production server or initializes database tables

import argparse
import uvicorn
from db.db import Base, engine
from db import models


def create_tables():
    """Ensure database tables and vector extensions are created."""
    Base.metadata.create_all(bind=engine)
    print("Database tables validated.")


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Launches the production FastAPI application using Uvicorn."""
    print(f"Starting Medijinn API server at http://{host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medijinn Clinical AI Platform")
    parser.add_argument("--init-db", action="store_true", help="Initialize database tables and exit")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    create_tables()

    if not args.init_db:
        start_server(host=args.host, port=args.port, reload=not args.no_reload)