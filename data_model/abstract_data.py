# abstract data class

from abc import ABC, abstractmethod

import pandas as pd

from config import config


class AbstractData(ABC):
    def __init__(self):
        input_file = config.input_file(self.FILE_NAME)
        self.data = pd.read_csv(input_file)

    @property
    @abstractmethod
    def FILE_NAME(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_label(self):
        pass
