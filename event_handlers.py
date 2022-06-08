import json

import requests
from fastapi_utils.tasks import repeat_every
from loguru import logger

from config import firebase_settings, interval_settings, worker_settings


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
