import numpy as np
import matplotlib.pyplot as plt

# Datos de 10 personas -> [peso, altura]
"""
Los datos normalizados para la altura son 2:1 donde 1.0 es igual a 2 metros,
1.8 / 2 para una persona de 1.8mts, rango 0-2mts
Y para el peso es lo mismo donde 200:1 0.1 son 20kg 0.2/2, rango 0-200kg
"""

personas = np.array([[0.75, 0.2], [0.92, 0.29],
                     [0.9, 0.37], [0.84, 0.33],
                     [0.8, 0.27], [0.77, .35],
                     [0.81, 0.7], [0.82, 0.6],
                     [0.92, 0.7], [0.75, 0.58]])

# 1 : delgado    0 : obesidad

clases = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Gráfica de dispersión (edad, ahorro)
plt.figure(figsize=(7, 7))
plt.title("Clasificación de obesidad", fontsize=20)
plt.scatter(personas[clases == 0].T[0],
            personas[clases == 0].T[1],
            marker="x", s=180, color="red",
            linewidths=5, label="Obesidad")
plt.scatter(personas[clases == 1].T[0],
            personas[clases == 1].T[1],
            marker="o", s=180, color="blue",
            linewidths=5, label="Delgado")
plt.xlabel("Estatura", fontsize=15)
plt.ylabel("Peso", fontsize=15)
plt.legend(bbox_to_anchor=(1.3, 0.15))
plt.box(False)
plt.xlim((0, 1.01))
plt.ylim((0, 1.01))
plt.grid()
plt.show()

"""

Función de activación
z = w0*x0 + w1*x1 + ⋯ + wn*xn

"""


def activacion(pesos, X, b):  # Función de activación binaria
    z = pesos * X  # Esta es la multiplicación de los pesos por las
    # características
    if z.sum() + b > 0:
        return 1
    else:
        return 0


pesos = np.random.uniform(-1, 1, size=len(personas[0]))
b = np.random.uniform(-1, 1)  # bias
tasa_de_aprendizaje = 0.01
epocas = 200

for _ in range(epocas):
    error_total = 0
    for i in range(len(personas)):
        prediccion = activacion(pesos, personas[i], b)
        error = clases[i] - prediccion
        error_total += error**2
        pesos[0] += tasa_de_aprendizaje * personas[i][0] * error
        pesos[1] += tasa_de_aprendizaje * personas[i][1] * error
        b += tasa_de_aprendizaje * error
    print(error_total, end=" ")
    # if error_total == 0: break;

print("\nUna persona de 58 kg y 1.66mts está: ", "Delgada"
      if activacion(pesos, [0.29, 0.83], b) == 0 else "Gorda")

plt.figure(figsize=(5, 3), dpi=200)
plt.title("Clasificación de obesidad", fontsize=10)

plt.scatter(personas[clases == 0].T[0],
            personas[clases == 0].T[1],
            marker="x", s=160, color="red",
            linewidths=3, label="Obeso")

plt.scatter(personas[clases == 1].T[0],
            personas[clases == 1].T[1],
            marker="o", s=160, color="blue",
            linewidths=3, label="Delgado")

for edad in np.arange(0, 1, 0.05):
    for ahorro in np.arange(0, 1, 0.05):
        color = activacion(pesos, [edad, ahorro], b)
        if color == 1:
            plt.scatter(edad, ahorro, marker="s", s=90,
                        color="blue", alpha=0.2, linewidths=0)
        else:
            plt.scatter(edad, ahorro, marker="s", s=90,
                        color="red", alpha=0.2, linewidths=0)

plt.xlabel("Estatura", fontsize=10)
plt.ylabel("Peso", fontsize=10)
plt.legend(bbox_to_anchor=(1.3, 0.15))
plt.box(False)
plt.xlim((0, 1.01))
plt.ylim((0, 1.01))
plt.show()
