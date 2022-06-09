from fastapi import FastAPI

from event_handlers import get_task, load_model, register_worker_handler, update_helath
from settings import app_settings


def get_app() -> FastAPI:
    """
    Return an Initialized FastAPI App.
    """
    fast_api_app = FastAPI(
        title=app_settings.app_name, version=app_settings.app_version, debug=app_settings.is_debug
    )
    fast_api_app.add_event_handler("startup", register_worker_handler)
    fast_api_app.add_event_handler("startup", load_model(fast_api_app))
    fast_api_app.add_event_handler("startup", update_helath)
    fast_api_app.add_event_handler("startup", get_task(fast_api_app))

    return fast_api_app


app = get_app()
