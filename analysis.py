import numpy as np
import squircle
from pathlib import Path
from squircle import Squircle
import matplotlib.pyplot as plt

SQUIRCLE_DATA_FOLDER = Path("data/magnetic-data")


def amount_of_domains(magnetic_data:np.ndarray):
    return np.sum(magnetic_data)

def deviation_to_vortex(magnetic_data:np.ndarray):
    return np.sum(magnetic_data)

squircles = squircle.load_squircles(str(SQUIRCLE_DATA_FOLDER))
squircles.sort(key=lambda squircle_object:squircle_object.width_nm)

data: dict[int, list[Squircle]] = {}

for squircle_object in squircles:
    if not squircle_object.width_nm in data:
        data[squircle_object.width_nm] = []
    data[squircle_object.width_nm].append(squircle_object)


def plot_squircle_function(func, name):
    fig, ax = plt.subplots()

    for width_nm in data:
        x = np.arange(6)/5
        y = np.zeros(6)
        count = np.zeros(6)

        grouped_squircles = data[width_nm]

        for squircle_object in grouped_squircles:
            y[squircle_object.squircle_factor] += func(squircle_object.data_masked)
            count[squircle_object.squircle_factor] += 1

        y /= count

        ax.plot(x,y,label=f"width nm:{width_nm}")


    ax.set_xlabel("Squircle interpolation factor")
    ax.set_ylabel(name)
    fig.legend()
    fig.savefig("plot.png")


plot_squircle_function(amount_of_domains, "total magnetism [M]")
