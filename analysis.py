import numpy as np
import squircle
from pathlib import Path
from squircle import Squircle
import matplotlib.pyplot as plt
from angle_calculator import compute_angle_difference
from numpy.fft import fft, fftfreq


SQUIRCLE_DATA_FOLDER = Path("data/magnetic-data")


def amount_of_domains(magnetic_data:np.ndarray):
    return np.sum(magnetic_data)

def deviation_to_vortex(magnetic_data:np.ndarray):
    plot_angles, plot_values, rms_angle, dangle_map, centers = compute_angle_difference(magnetic_data, real_data=True)
    return np.sum(np.abs(plot_angles))
    #return np.sum(magnetic_data)


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


def get_frequency_signal(signal, length):
    N = len(signal)
    T = length / N

    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]

    frequency = xf[np.argmax(np.abs(yf))]

    return frequency


if __name__ == "__main__":
    plot_squircle_function(deviation_to_vortex, "Deviation to vortex")


