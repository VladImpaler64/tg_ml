import csv


class PreData():
    def __init__(self, path: str):
        self.path = path

    def parse_csv(self):

        train_data: list
        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            train_data = list(reader)
            for i, row in enumerate(train_data):
                for j, x in enumerate(row):
                    train_data[i][j] = int(x)
        self.data_ = train_data
        return self.prepare()

    def prepare(self) -> tuple:

        len_train = int(len(self.data_)*0.9)
        train = self.data_[0:len_train]  # Crea una shallow copy
        rest = self.data_[len_train:]
        # Get ŷ
        len_each = len(self.data_[0]) - 1
        y = []
        for row in self.data_:
            y.append(row[len_each])

        return (train, rest, y)
