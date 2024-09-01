import numpy as np
from src.train_data import PreData


class Perceptron(object):
    # Si hay errores y estos suben debes bajar el eta
    # Y no se podría controlar en tiempo de ejecución el eta?
    def __init__(self, eta=0.01, epocas=2500, random_state=51):  # 51
        self.eta = eta
        self.epocas = epocas
        self.random_state = random_state

    def fit(self, X, y):
        """
        Training data
        """
        rgen = np.random.RandomState(self.random_state)
        # X.shape gets the number of characteristics of instances Xi + bias
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        """
        loc is the center of the normal distribution
        scale is how spread the curve will be or flat
        size is the returned array size (x, y)
        """
        self.errors_ = []

        for _ in range(self.epocas):
            errors = 0
            for xi, target in zip(X, y):
                # wj = : wj + Awj
                # Awj = n(yi + ŷi)xji
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi  # For each sample
                self.w_[0] += update  # Only for bias weight
                errors += int(update != 0.0)
            self.errors_.append(errors)
        print(self.errors_)
        return self  # Esto se usa para method cascading, una ref

    def net_input(self, X):
        # Producto escalar vectorial + bias
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # Función escalón unitario
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def predict_mult(self, X):
        """
        Lo ideal es que la cambie por una función sigmoide, creo
        """
        return self.net_input(X)  # Esta será para que me
        # arroje probabilidad para el OvA, multiclase

    def feed(self):
        # Feed from trained weights
        # self.w_ =
        return


"""
Entrenar multiclase
"""


def multi_fit(datos: tuple, class_number: int):

    all_perceptrons = []
    for _ in range(class_number):  # A perceptron per class
        all_perceptrons.append(Perceptron())

    y = datos[2][0:len(datos[0])]  # Only the training data

    for i, characteristic in enumerate(all_perceptrons):
        fix_y = [-1 for _ in range(len(y))]
        for j, val in enumerate(y):
            if val == i + 1:
                fix_y[j] = 1

        characteristic.fit(np.array(datos[0]), np.array(fix_y))

    return all_perceptrons


def channel_classification(entrada, perceptrones,
                           class_names, save_flag=False) -> str:
    result = []
    print(f'Clasificando la entrada {entrada}')
    for perceptron in perceptrones:
        result.append(perceptron.predict_mult(entrada))

    max_p = result[0]
    index = 0

    # Most probable
    for i, val in enumerate(result):
        if val > max_p:
            max_p = val
            index = i

    print(result)
    if save_flag:
        save_w(perceptrones)

    return class_names[index + 1]


def save_w(perceptrones: list):
    if input("¿Guardamos los pesos? (yes/no)") == "yes":
        for i, per in enumerate(perceptrones):
            np.save(f'./pretrained/pretrained{i}.npy', per.w_)


la_data = PreData("./tg_channels.csv").parse_csv()
classes = {
    1: "Excelente",
    2: "Bueno",
    3: "Regular",
    4: "Malo",
    5: "Deficiente",
}


def parse_input(text: str):
    numbered = [int(x) for x in text.split(",")]
    numbered.append(0)
    return numbered


def start():
    perceptrones = []

    for i in range(5):
        nuevo = Perceptron()
        nuevo.w_ = np.load(f'./pretrained/pretrained{i}.npy')
        perceptrones.append(nuevo)

    # Haz la lógica para que se entre desde stdin
    parsed_input = parse_input(input(f'Make entry, comma separated, example: 200,23,15,6,54,14,2,23,3,1,0,3,4,0\n\n Categories: Channel Subs, Total Comments, Congruent Comments, Distinct Users, Word Count, Hour(24), Minutes, Positive Reactions, Negative Reactions, Actual Theme, Poll, Stars, Post Type, Hashtag:\n\n> '))
    print(f'\nClassification for this post: {channel_classification(parsed_input, perceptrones, classes)}')


start()
