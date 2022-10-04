import os
from typing import Callable

import dill

from config import config
from util.logger import log


class FileIo:
    @staticmethod
    def load_model(file_name: str):
        pickle_path = os.path.join(config.cache_dir(), file_name)
        log.info(f"Loading {file_name}")

        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"File not found: {file_name}")

        with open(pickle_path, 'rb') as handle:
            return dill.load(handle)

    @staticmethod
    def save_model(file_name: str, obj):
        log.info(f"Creating {file_name}")
        pickle_path = os.path.join(config.cache_dir(), file_name)
        with open(pickle_path, 'wb') as handle:
            dill.dump(obj, handle)
