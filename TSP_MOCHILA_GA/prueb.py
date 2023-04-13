import matplotlib.pyplot as plt
import numpy as np

# Crea una figura con 2 subplots en una columna
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Genera algunos datos aleatorios para graficar
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Grafica los datos en cada subplot
axs[0].plot(x, y1)
axs[0].set_title('Gráfica 1')
axs[1].plot(x, y2)
axs[1].set_title('Gráfica 2')

# Muestra la ventana y permite la paginación
plt.show()