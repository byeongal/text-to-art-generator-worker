from pydantic import BaseSettings, Field


class AppSettings(BaseSettings):
    """
    Settings about App
    """

    app_name: str = "Text To Art Generator Worker"
    app_version: str = "0.1.0-dev"
    api_prefix: str = "/api"
    is_debug: bool = True


class IntervalSettings(BaseSettings):
    """
    Settings about interval
    """

    task: int = Field(default=60, description="")
    health: int = Field(default=10, description="")


class WorkerSettings(BaseSettings):
    """
    Settings aboout Worker
    """

    worker_id: str = Field(default=None, description="")
    worker_key: str = Field(default=None, description="")


class FirebaseSettings(BaseSettings):
    """
    Settings about firebase
    """

    func_url: str


app_settings = AppSettings()
interval_settings = IntervalSettings()
worker_settings = WorkerSettings()
firebase_settings = FirebaseSettings()
