from config import config


class OutputWriter:
    """
    Output writer for Toxic Comment Classification Challenge
    """

    def __init__(self, filename):
        self.file = open(config.result_file(filename), 'w')

    def write(self, ids, Y):
        self.file.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
        for i in range(0, len(ids)):
            self._write(ids[i], Y[i])
        self.close()

    def _write(self, id, prediction):
        self.file.write(id + ',')
        for i in range(1, len(prediction)):
            self.file.write(str(prediction[i]))
            if i != len(prediction) - 1:
                self.file.write(',')
        self.file.write('\n')

    def close(self):
        self.file.close()
