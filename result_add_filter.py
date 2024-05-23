import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# with open(
#     "D:/Python_project/1_2_MAGA/Digital electronics modulator/data.txt", "r"
# ) as file:
#     tuning_sequence_str = file.read().splitlines()[0]
# tuning_sequence = np.array([int(bit) for bit in tuning_sequence_str.split()])

# Загрузка исходных данных из файла (сигнала)
bpsk_signal = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/bpsk_signal_var2.txt"
)
# Длина одного бита в отсчётах
bit_length = 250


# Функция для создания фильтра нижних частот
def butter_lowpass(cutoff, fs, order=5):
    # Частота Найквиста
    nyq = 0.5 * fs
    # Нормализация: частота среза к частоте Найквиста
    normal_cutoff = cutoff / nyq
    # order - порядок фильтра, btype="low" - фильтр нижних частот
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Функция для применения фильтра нижних частот к сигналу
def lowpass_filter(data, cutoff, fs, order=5):
    # Коэф для фильтра (определяют его характеристики)
    b, a = butter_lowpass(cutoff, fs, order=order)
    # Линейная фильтрация данных с использованием разностного уравнения. Выход: последовательность
    y = lfilter(b, a, data)
    return y


# Реализация схемы Костаса для демодуляции исходного сигнала
def costas_loop(signal, Kp, Ki):
    N = len(signal)
    phase_error = np.zeros(N)
    integrator = np.zeros(N)
    phase_est = np.zeros(N)
    freq_est = np.zeros(N)
    phase = 0
    freq = 0

    for i in range(1, N):
        # Интегратор (накапливает ошибку фазы)
        integrator[i] = integrator[i - 1] + phase_error[i - 1]
        # Оценка фазы
        phase_est[i] = (
            phase_est[i - 1] + freq + Kp * phase_error[i] + Ki * integrator[i]
        )
        # Текущая фаза (остаток от деления на 2pi)
        phase = np.mod(phase_est[i], 2 * np.pi)
        # Текущая оценка частоты
        freq_est[i] = freq
        # Квадратурная компонента
        I = signal[i] * np.cos(phase)
        # Фазовая компонента
        Q = signal[i] * np.sin(phase)
        # Ошибка фазы на текущем шаге
        phase_error[i] = I * Q
        # Оценка частоты
        freq += Kp * phase_error[i]

    return phase_est


# Демодуляция
def demodulate_bpsk(signal, bit_length):
    signal_length = len(signal)
    num_bits = signal_length // bit_length
    # Разбиение сигнала на отдельные биты
    bits = np.zeros(num_bits, dtype=int)
    # Коэф-ты усиления пропорциональной и интегральной обратной связи
    Kp = 0.1
    Ki = 0.01
    sampling_freq = 1
    phase_est = costas_loop(signal, Kp, Ki)

    for i in range(num_bits):
        bit_signal = signal[i * bit_length : (i + 1) * bit_length]
        I = bit_signal * np.cos(phase_est[i * bit_length : (i + 1) * bit_length])
        Q = bit_signal * np.sin(phase_est[i * bit_length : (i + 1) * bit_length])
        I_filtered = lowpass_filter(I, 0.1, sampling_freq)
        # Определение бита по среднему значению
        bits[i] = int(np.mean(I_filtered) < 0)
    bits = 1 - bits

    return bits


# Демодуляция BPSK сигнала
recovered_bits = demodulate_bpsk(bpsk_signal[:, 1], bit_length)
# Загрузка эталонной последовательности
data_package = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data_package_var2.txt",
    dtype=int,
)
# Сравнение демодулированной последовательности с эталонной
if len(recovered_bits) != len(data_package):
    print("Разное количество битов! Проверка не будет выполнена!")
else:
    correct_bits = np.sum(recovered_bits == data_package)
    total_bits = len(data_package)
    accuracy = correct_bits / total_bits * 100
    print(f"Точность: {accuracy:.2f}%")
# ПСП (псевдослучайная последовательность)
psp = np.array([0, 0, 1, 0, 1, 1, 1])
# Корреляция восстановленной битовой последовательности с ПСП
psp_length = len(psp)
correlation = np.correlate(recovered_bits, psp, mode="valid")
start_index = np.argmax(correlation)
# Извлечение информационных бит
informational_bits = recovered_bits[
    start_index + psp_length : start_index + psp_length + len(data_package) - psp_length
]
# Вывод восстановленной битовой последовательности и ее длины
print(recovered_bits)
print(len(recovered_bits))
# Вывод информационных битов и их длины
print("Информационные биты:", informational_bits)
print("Длина информационных битов:", len(informational_bits))

# Визуализация
plt.figure(figsize=(10, 5))

# Исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(bpsk_signal[:, 0], bpsk_signal[:, 1])
plt.title("Исходный BPSK сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

# Восстановленная битовая последовательность
plt.subplot(3, 1, 2)
plt.stem(recovered_bits, use_line_collection=True)
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

# Информационные биты
plt.subplot(3, 1, 3)
plt.stem(informational_bits, use_line_collection=True)
plt.title("Информационные биты")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
