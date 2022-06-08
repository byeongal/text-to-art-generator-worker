import json
import sys

import requests
from fastapi_utils.tasks import repeat_every
from loguru import logger

from config import firebase_settings, interval_settings, worker_settings
from enums import ExitStatusEnum


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
                    logger.info(
                        f"Successfully verified permissions for this worker(${worker_settings.worker_id})."
                    )
                else:
                    logger.error(
                        f"Failed to verify permissions for this worker(${worker_settings.worker_id})."
                    )
                    sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except requests.RequestException as error:
            logger.error(f"Request Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except Exception as error:
            logger.error(f"Unknown Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)

    else:
        logger.info("Register New Worker")
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
                logger.info(
                    f"Worker ID: {worker_settings.worker_id} Worker Key: {worker_settings.worker_key}"
                )
            else:
                logger.error(f"Worker Register Error : {response}")
                sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except requests.RequestException as error:
            logger.error(f"Request Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)
        except Exception as error:
            logger.error(f"Unknown Error : {error}")
            sys.exit(ExitStatusEnum.REGISTER_WORKER_ERROR)


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
            logger.info(f"Update Health {worker_settings.worker_id}")
        else:
            logger.error(f"Error Update Worker{worker_settings.worker_id} Status")
    except requests.RequestException as error:
        logger.error(f"Request Error : {error}")
    except Exception as error:
        logger.error(f"Unknown Error : {error}")
