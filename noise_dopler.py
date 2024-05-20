import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
fs = 100  # Частота дискретизации
Tb = 1  # Длительность одного бита
f0 = 10  # Частота несущей
N = 25  # Число битов в сообщении
doppler_shift = 2  # Доплеровский сдвиг
snr = 10  # Отношение сигнал/шум (в дБ)

# Генерация случайной битовой последовательности
bits = np.random.randint(0, 2, N)

# Временная шкала
t = np.linspace(0, N * Tb, N * fs, endpoint=False)

# Генерация ФМ-2 сигнала с доплеровским сдвигом
modulated_signal = np.cos(
    2 * np.pi * (f0 + doppler_shift) * t + np.pi * bits.repeat(fs * Tb)
)

# Добавление шума к сигналу
signal_power = np.mean(modulated_signal**2)
noise_power = signal_power / (10 ** (snr / 10))
noise = np.sqrt(noise_power) * np.random.normal(size=modulated_signal.shape)
noisy_signal = modulated_signal + noise

# Демодуляция ФМ-2 сигнала
I = np.cos(2 * np.pi * (f0 + doppler_shift) * t) * noisy_signal
Q = np.sin(2 * np.pi * (f0 + doppler_shift) * t) * noisy_signal

# Применение фильтра низких частот
I_filt = np.convolve(I, np.ones(fs * Tb) / fs * Tb, mode="same")
Q_filt = np.convolve(Q, np.ones(fs * Tb) / fs * Tb, mode="same")

# Восстановление битовой последовательности
recovered_bits = I_filt[fs * Tb // 2 :: fs * Tb] > 0

# Графики
plt.figure(figsize=(12, 12))

# Исходный сигнал
plt.subplot(4, 1, 1)
plt.plot(t, modulated_signal)
plt.title("ФМ-2 сигнал с доплеровским сдвигом")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")

# Сигнал с шумом
plt.subplot(4, 1, 2)
plt.plot(t, noisy_signal)
plt.title("ФМ-2 сигнал с шумом")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")

# I и Q компоненты
plt.subplot(4, 1, 3)
plt.plot(t, I_filt, label="I компонент")
plt.plot(t, Q_filt, label="Q компонент")
plt.title("I и Q компоненты")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.legend()

# Восстановленная битовая последовательность
plt.subplot(4, 1, 4)
plt.step(np.arange(N), recovered_bits, where="mid")
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Номер бита")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
