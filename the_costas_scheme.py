import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
Fs = 1000  # Частота дискретизации
T = 1  # Длительность сигнала в секундах
t = np.linspace(0, T, T * Fs, endpoint=False)  # Временные отсчеты

# Несущая частота сигнала
fc = 10  # Частота несущей сигнала

# Модулированный сигнал (для примера, использование фазовой модуляции)
kf = 1.0  # Коэффициент частотной модуляции
phi = 0.5 * np.cos(2 * np.pi * 2 * t)  # Фазовая модуляция с частотой 2 Гц

# Модулированный сигнал
modulated_signal = np.cos(2 * np.pi * (fc + kf * phi) * t)

# Демодуляция сигнала с помощью контура Костаса
cos_signal = np.cos(2 * np.pi * fc * t)
sin_signal = np.sin(2 * np.pi * fc * t)

I = modulated_signal * cos_signal
Q = modulated_signal * sin_signal

# Фазовая ошибка
phase_error = np.arctan2(np.sum(Q), np.sum(I))

# Восстановленная несущая частота
recovered_fc = fc + (phase_error / (2 * np.pi * T))

print("Исходная несущая частота:", fc)
print("Восстановленная несущая частота:", recovered_fc)

# Визуализация результатов
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, phi)
plt.title("Фазовая модуляция")

plt.subplot(3, 1, 2)
plt.plot(t, modulated_signal)
plt.title("Модулированный сигнал")

plt.subplot(3, 1, 3)
plt.plot(t, I, label="I (In-phase)")
plt.plot(t, Q, label="Q (Quadrature)")
plt.title("Выходы контура Костаса")
plt.legend()

plt.tight_layout()
plt.show()
