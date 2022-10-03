# abstract data class

from abc import ABC, abstractmethod

import pandas as pd

from config import config
from util.downloader import Downloader


class AbstractData(ABC):
    def __init__(self):
        file = self._load_file()
        self.data = self._preprocess_data(file)

    def _load_file(self):
        if self.FILE_URL:
            input_file = Downloader().download(self.FILE_NAME, self.FILE_URL)
        else:
            input_file = config.input_file(self.FILE_NAME)
        return input_file

    def _preprocess_data(self, file):
        return pd.read_csv(file)

    @property
    @abstractmethod
    def FILE_NAME(self):
        pass

    @property
    @abstractmethod
    def FILE_URL(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_label(self):
        pass
