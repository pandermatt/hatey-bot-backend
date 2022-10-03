import os

from config import config
from util.logger import log


class Downloader:
    @staticmethod
    def download(file_name, url, zip=False):
        cache_dr = config.cache_dir()
        local_path = os.path.join(cache_dr, file_name)
        if not os.path.exists(local_path):
            import dload
            if zip:
                dload.save_unzip(url, cache_dr)
            else:
                dload.save(url, local_path)
            log.info(f"Downloading finished: {url}")
        return local_path
