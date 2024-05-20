import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
fs = 100  # Частота дискретизации
Tb = 1  # Длительность одного бита
f0 = 5  # Частота несущей
N = 40  # Число битов в сообщении

# Генерация случайной битовой последовательности
bits = np.random.randint(0, 2, N)

# Генерация ФМ-2 сигнала
t = np.linspace(0, N * Tb, N * fs, endpoint=False)
modulated_signal = np.cos(2 * np.pi * f0 * t + np.pi * bits.repeat(fs * Tb))

# Демодуляция ФМ-2 сигнала
I = np.cos(2 * np.pi * f0 * t) * modulated_signal
Q = np.sin(2 * np.pi * f0 * t) * modulated_signal

# Применение фильтра низких частот
I_filt = np.convolve(I, np.ones(fs * Tb) / fs * Tb, mode="same")
Q_filt = np.convolve(Q, np.ones(fs * Tb) / fs * Tb, mode="same")

# Восстановление битовой последовательности
recovered_bits = I_filt[fs * Tb // 2 :: fs * Tb] > 0

# Графики
plt.figure(figsize=(12, 10))

# Исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(t, modulated_signal)
plt.title("ФМ-2 сигнал")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")

# I и Q компоненты
plt.subplot(3, 1, 2)
plt.plot(t, I, label="I компонент")
plt.plot(t, Q, label="Q компонент")
plt.title("I и Q компоненты")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.legend()

# Восстановленная битовая последовательность
plt.subplot(3, 1, 3)
plt.step(np.arange(N), recovered_bits, where="mid")
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Номер бита")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
