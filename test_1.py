import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian, convolve

# Параметры сигнала
bit_rate = 25  # Скорость передачи битов в битах в секунду
duration = 1.0  # Длительность сигнала в секундах
fs = 20 * bit_rate  # Частота дискретизации (предполагаемая)
f_carrier = 20  # Частота несущей синусоиды в Гц

# Генерируем последовательность битов
num_bits = int(bit_rate * duration)
bits = np.random.randint(0, 2, num_bits)

# Генерируем временную ось
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Генерируем несущую синусоиду
carrier_wave = np.sin(2 * np.pi * f_carrier * t)

# Генерируем модулированный сигнал
modulated_signal = np.zeros_like(carrier_wave)

bit_period_samples = int(fs / bit_rate)  # Число сэмплов в одном периоде бита

for i, bit in enumerate(bits):
    phase_shift = np.pi if bit == 1 else 0  # Фазовый сдвиг на 180 градусов при бите 1
    modulated_signal[i * bit_period_samples : (i + 1) * bit_period_samples] = np.sin(
        2 * np.pi * f_carrier * t[i * bit_period_samples : (i + 1) * bit_period_samples]
        + phase_shift
    )

# Добавляем шум
SNR_dB = 10  # Отношение сигнал/шум в децибелах
signal_power = np.sum(modulated_signal**2) / len(modulated_signal)
noise_power = signal_power / (10 ** (SNR_dB / 10))
noise = np.random.normal(0, np.sqrt(noise_power), modulated_signal.shape)
noisy_signal = modulated_signal + noise

# Моделируем эффект доплеровского сдвига
doppler_factor = 2.0  # Коэффициент доплеровского сдвига
doppler_shift = np.cos(
    2 * np.pi * doppler_factor * t
)  # Модулирующая функция для доплеровского сдвига
doppler_signal = noisy_signal * doppler_shift

# Строим график сигналов
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, carrier_wave)
plt.title("Несущая синусоида")

plt.subplot(4, 1, 2)
plt.plot(t, bits.repeat(bit_period_samples))
plt.title("Последовательность бит")

plt.subplot(4, 1, 3)
plt.plot(t, noisy_signal)
plt.title("Шумный BPSK сигнал")

plt.subplot(4, 1, 4)
plt.plot(t, doppler_signal)
plt.title("BPSK сигнал с доплеровским сдвигом")

plt.tight_layout()
plt.show()
