import json
import sys
from typing import Callable

import loguru
import requests
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

from constants import ExitStatusEnum, TaskStatusEnum
from settings import firebase_settings, interval_settings, worker_settings
from utils.inference import generate_image
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


def get_task(app: FastAPI) -> Callable:
    """
    Event Handler to get task from Firebase
    """

    @repeat_every(seconds=interval_settings.task)
    def interval() -> None:
        try:
            response = requests.post(
                f"{firebase_settings.func_url}/getWorkerStatus",
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
            if not (response.status_code == 200 and "result" in response.json()):
                loguru.logger.error(f"Error Get Worker{worker_settings.worker_id} Status")
                return
            result = response.json()["result"]
            if not "runningTaskId" in result:
                loguru.logger.info("This worker is not currently assigned a task.")
                return
            running_task_id = result["runningTaskId"]
            response = requests.post(
                f"{firebase_settings.func_url}/getTaskStatus",
                headers={"accept": "application/json", "Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "data": {
                            "taskId": running_task_id,
                        }
                    }
                ),
            )
            if not (response.status_code == 200 and "result" in response.json()):
                loguru.logger.error(f"Error Get Tast{running_task_id} Status")
                return
            result = response.json()["result"]
            if result["status"] != TaskStatusEnum.ASSIGNED.value:
                loguru.logger.info(f"Task({running_task_id}) is already working on it.")
                return
            loguru.logger.info(f"Task({running_task_id}) Test")
            generate_image(
                app=app,
                text_prompt=result["input"]["text"],
                seed=result["input"]["seed"],
                task_id=running_task_id,
            )
        except Exception as err:
            loguru.logger.error(f"Server Error : {err}")

    return interval
