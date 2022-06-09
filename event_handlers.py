import json
import sys
from typing import Callable

import loguru
import requests
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

from constants import ExitStatusEnum
from settings import firebase_settings, interval_settings, worker_settings
from utils.model import (
    get_normalize,
    load_clip_model,
    load_diffusion,
    load_diffusion_model,
    load_lips,
)


def register_worker_handler() -> None:
    """
    Event Handelr to register Inference Worker
    """
    if worker_settings.worker_id is not None and worker_settings.worker_key is not None:
        try:
            response = requests.post(
                f"{firebase_settings.func_url}/vertifyWorker",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "data": {
                            "workerId": worker_settings.worker_id,
                            "workerKey": worker_settings.worker_key,
                        }
                    }
                ),
            )
            if response.status_code == 200 and "result" in response.json():
                result = response.json()["result"]
                if result:
                    loguru.logger.info(
                        f"Successfully verified permissions for this worker(${worker_settings.worker_id})."
                    )
                else:
                    loguru.logger.error(
                        f"Failed to verify permissions for this worker(${worker_settings.worker_id})."
                    )
                    sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except requests.RequestException as error:
            loguru.logger.error(f"Request Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except Exception as error:
            loguru.logger.error(f"Unknown Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)

    else:
        loguru.logger.info("Register New Worker")
        try:
            response = requests.post(
                f"{firebase_settings.func_url}/registerWorker",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                data=json.dumps({"data": {}}),
            )
            if response.status_code == 200 and "result" in response.json():
                result = response.json()["result"]
                worker_settings.worker_id = result["workerId"]
                worker_settings.worker_key = result["workerKey"]
                loguru.logger.info(
                    f"Worker ID: {worker_settings.worker_id} Worker Key: {worker_settings.worker_key}"
                )
            else:
                loguru.logger.error(f"Worker Register Error : {response}")
                sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except requests.RequestException as error:
            loguru.logger.error(f"Request Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except Exception as error:
            loguru.logger.error(f"Unknown Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)


def load_model(app: FastAPI) -> Callable:
    """
    Event Handler to load Model.
    """

    def inner_func() -> None:
        loguru.logger.info("Load Model")
        app.state.normalize = get_normalize()
        app.state.clip_models = load_clip_model()
        app.state.lpips_model = load_lips()
        app.state.diffusion_model = load_diffusion_model()
        app.state.diffusion = load_diffusion()

    return inner_func


@repeat_every(seconds=interval_settings.health)
def update_helath() -> None:
    """
    Event Handler to update worker status
    """
    try:
        response = requests.post(
            f"{firebase_settings.func_url}/updateWorkerStatus",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(
                {
                    "data": {
                        "workerId": worker_settings.worker_id,
                        "workerKey": worker_settings.worker_key,
                    }
                }
            ),
        )
        if response.status_code == 200 and "result" in response.json():
            loguru.logger.info(f"Update Health {worker_settings.worker_id}")
        else:
            loguru.logger.error(f"Error Update Worker{worker_settings.worker_id} Status")
    except requests.RequestException as error:
        loguru.logger.error(f"Request Error : {error}")
    except Exception as error:
        loguru.logger.error(f"Unknown Error : {error}")
