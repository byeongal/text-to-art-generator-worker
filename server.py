from fastapi import FastAPI

from config import app_settings
from event_handlers import update_helath


def get_app() -> FastAPI:
    """
    Return an Initialized FastAPI App.
    """
    fast_api_app = FastAPI(
        title=app_settings.app_name, version=app_settings.app_version, debug=app_settings.is_debug
    )
    fast_api_app.add_event_handler("startup", update_helath)

    return fast_api_app


app = get_app()
