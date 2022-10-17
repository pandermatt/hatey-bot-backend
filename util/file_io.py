import os
from typing import Callable

import dill

from config import config
from util.logger import log


class FileIo:
    @staticmethod
    def load_obj(file_name: str):
        pickle_path = os.path.join(config.cache_dir(), file_name)
        log.info(f"Loading {file_name}")

        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"File not found: {file_name} at {pickle_path}")

        with open(pickle_path, 'rb') as handle:
            return dill.load(handle)

    @staticmethod
    def save_obj(file_name: str, obj):
        log.info(f"Creating {file_name}")
        pickle_path = os.path.join(config.cache_dir(), file_name)
        with open(pickle_path, 'wb') as handle:
            dill.dump(obj, handle)

    @staticmethod
    def cached(file_name: str, func: Callable):
        def wrapper(*args, **kwargs):
            try:
                return FileIo.load_obj(file_name)
            except FileNotFoundError:
                log.info(f"File not found: {file_name}, generating...")
                result = func(*args, **kwargs)
                FileIo.save_obj(file_name, result)
                return result

        return wrapper
