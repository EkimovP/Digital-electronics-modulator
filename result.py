import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


# Функция для загрузки данных из файлов
def load_data(filename):
    return np.loadtxt(filename)


# Функции для фильтрации
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Загрузка данных
training_sequence = load_data(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data_package.txt"
)
bpsk_signal_data = load_data(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/bpsk_signal.txt"
)

# Извлечение оси времени и BPSK сигнала
t = bpsk_signal_data[:, 0]
bpsk_signal = bpsk_signal_data[:, 1]

# Параметры сигнала
fs = 1 / (t[1] - t[0])  # Частота дискретизации
Tb = len(t) / fs  # Длительность одного бита (оценка)
f0 = 10  # Частота несущей (примерная)
cutoff = f0 / 2  # Частота среза фильтра

# Демодуляция BPSK сигнала с использованием схемы Костаса
I = np.cos(2 * np.pi * f0 * t) * bpsk_signal
Q = np.sin(2 * np.pi * f0 * t) * bpsk_signal

# Фильтрация I и Q компонентов
I_filt = lowpass_filter(I, cutoff, fs)
Q_filt = lowpass_filter(Q, cutoff, fs)

# Фазовая ошибка
phase_error = np.arctan2(Q_filt, I_filt)

# Коррекция фазы
corrected_signal = np.cos(2 * np.pi * f0 * t + phase_error)

# Восстановление битовой последовательности
recovered_bits = corrected_signal > 0

# Преобразование recovered_bits в 0 и 1
recovered_bits = recovered_bits.astype(int)

# Сохранение битовой последовательности в файл
np.savetxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/recovered_bits.txt",
    recovered_bits,
    fmt="%d",
)

# Графики
plt.figure(figsize=(12, 12))

# Исходный BPSK сигнал
plt.subplot(4, 1, 1)
plt.plot(t, bpsk_signal)
plt.title("BPSK сигнал")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")

# I и Q компоненты
plt.subplot(4, 1, 2)
plt.plot(t, I_filt, label="I компонент")
plt.plot(t, Q_filt, label="Q компонент")
plt.title("I и Q компоненты")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.legend()

# Корректированный сигнал
plt.subplot(4, 1, 3)
plt.plot(t, corrected_signal)
plt.title("Корректированный сигнал")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")

# Восстановленная битовая последовательность
plt.subplot(4, 1, 4)
plt.step(np.arange(len(recovered_bits)), recovered_bits, where="mid")
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Номер бита")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
