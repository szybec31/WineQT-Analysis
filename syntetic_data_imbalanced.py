# Generowanie wykresu danych niezbalansowanych na potrzeby prezentacji

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Klasa większościowa
majority = np.random.normal(
    loc=[0, 0],
    scale=[1.0, 1.0],
    size=(300, 2)
)

# Klasa mniejszościowa
minority = np.random.normal(
    loc=[1, 1],
    scale=[0.5, 0.5],
    size=(20, 2)
)

plt.figure(figsize=(7, 6))

plt.scatter(
    majority[:, 0],
    majority[:, 1],
    label="Majority class"
)

plt.scatter(
    minority[:, 0],
    minority[:, 1],
    label="Minority class"
)


plt.title("Rare majority example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


