import numpy as np
import matplotlib.pyplot as plt


class Modulator:
    def __init__(self, carrier_freq, sampling_rate):
        self.carrier_freq = carrier_freq
        self.sampling_rate = sampling_rate

    def modulate(self, signal):
        t = np.arange(len(signal)) / self.sampling_rate
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq * t)

        # Восстановление несущей частоты (Carrier Recovery)
        signal_with_carrier = signal * carrier

        # Применение фильтра нижних частот (Low Pass Filter)
        cutoff_freq = 0.2 * self.carrier_freq
        num_taps = int(0.1 * self.sampling_rate / cutoff_freq)
        filter_window = np.ones(num_taps) / num_taps
        filtered_signal = np.convolve(signal_with_carrier, filter_window, mode="same")

        # Пороговый элемент (Thresholding)
        thresholded_signal = np.where(np.real(filtered_signal) > 0, 1, 0)

        # Восстановление тактовой частоты (Clock Recovery)
        phase_diff = np.angle(filtered_signal[1:] * np.conj(filtered_signal[:-1]))
        phase_diff = np.unwrap(phase_diff)
        clock_rate = np.mean(phase_diff) / (2 * np.pi) * self.sampling_rate
        interpolated_clock = np.interp(
            np.arange(len(signal)), np.arange(len(phase_diff)), phase_diff
        )
        clock_signal = np.exp(-1j * interpolated_clock)

        # Извлечение бинарных данных с использованием тактового сигнала
        binary_data = thresholded_signal * clock_signal

        return binary_data


# Параметры сигнала и модулятора
carrier_freq = 10  # несущая частота (Гц)
sampling_rate = 500  # частота дискретизации (Гц)
duration = 1.0  # длительность сигнала в секундах

# Создание FM-2 модулированной синусоиды с шумом и эффектом доплера
np.random.seed(0)
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

fm_deviation = 5  # размах частотной модуляции
modulation_freq = 10  # частота модуляции
carrier_signal = np.sin(
    2 * np.pi * carrier_freq * t
    + fm_deviation * np.sin(2 * np.pi * modulation_freq * t)
)

# Добавление шума
noise_power = 0.01 * np.var(carrier_signal)  # мощность шума (10% от дисперсии сигнала)
noisy_signal = carrier_signal + np.random.normal(
    scale=np.sqrt(noise_power), size=len(carrier_signal)
)

# Создание экземпляра модулятора
modulator = Modulator(carrier_freq, sampling_rate)

# Извлечение бинарных данных из зашумленного сигнала
binary_data_output = modulator.modulate(noisy_signal)

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, carrier_signal, label="FM-2 Signal")
plt.title("FM-2 Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label="Noisy FM-2 Signal")
plt.title("FM-2 Signal with Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, binary_data_output.real, label="Binary Data Output")
plt.title("Extracted Binary Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.ylim([-1.2, 1.2])
plt.legend()
print(binary_data_output.real)

plt.tight_layout()
plt.show()
