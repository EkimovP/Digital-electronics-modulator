import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Загрузка данных из файлов
with open(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data.txt", "r"
) as file:
    tuning_sequence_str = file.read().splitlines()[0]

tuning_sequence = np.array([int(bit) for bit in tuning_sequence_str.split()])

bpsk_signal = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/bpsk_signal.txt"
)

# Длина одного бита в отсчётах
bit_length = 250


# Функция для демодуляции BPSK сигнала с использованием схемы Костаса
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def costas_loop(signal, Kp, Ki):
    N = len(signal)
    phase_error = np.zeros(N)
    integrator = np.zeros(N)
    phase_est = np.zeros(N)
    freq_est = np.zeros(N)
    phase = 0
    freq = 0

    for i in range(1, N):
        integrator[i] = integrator[i - 1] + phase_error[i - 1]
        phase_est[i] = (
            phase_est[i - 1] + freq + Kp * phase_error[i] + Ki * integrator[i]
        )
        phase = np.mod(phase_est[i], 2 * np.pi)
        freq_est[i] = freq
        I = signal[i] * np.cos(phase)
        Q = signal[i] * np.sin(phase)
        phase_error[i] = I * Q

        freq += Kp * phase_error[i]

    return phase_est


def demodulate_bpsk(signal, bit_length):
    signal_length = len(signal)
    num_bits = signal_length // bit_length

    # Разбиение сигнала на отдельные биты
    bits = np.zeros(num_bits, dtype=int)
    Kp = 0.1
    Ki = 0.01
    sampling_freq = 1  # Assuming normalized frequency
    phase_est = costas_loop(signal, Kp, Ki)

    for i in range(num_bits):
        bit_signal = signal[i * bit_length : (i + 1) * bit_length]
        # Демодуляция Костаса
        I = bit_signal * np.cos(phase_est[i * bit_length : (i + 1) * bit_length])
        Q = bit_signal * np.sin(phase_est[i * bit_length : (i + 1) * bit_length])
        I_filtered = lowpass_filter(I, 0.1, sampling_freq)
        bits[i] = int(np.mean(I_filtered) < 0)
    bits = 1 - bits

    return bits


# Демодуляция BPSK сигнала
recovered_bits = demodulate_bpsk(bpsk_signal[:, 1], bit_length)

data_package = np.loadtxt(
    "D:/Python_project/1_2_MAGA/Digital electronics modulator/data_package.txt",
    dtype=int,
)

# Сравнение демодулированной последовательности с эталонной
correct_bits = np.sum(recovered_bits == data_package)
total_bits = len(data_package)
accuracy = correct_bits / total_bits * 100

print(f"Точность: {accuracy:.2f}%")

print(recovered_bits)
print(len(recovered_bits))

# Визуализация
plt.figure(figsize=(10, 5))

# Исходный сигнал
plt.subplot(2, 1, 1)
plt.plot(bpsk_signal[:, 0], bpsk_signal[:, 1])
plt.title("Исходный BPSK сигнал")
plt.xlabel("Время")
plt.ylabel("Амплитуда")

# Восстановленная битовая последовательность
plt.subplot(2, 1, 2)
plt.stem(recovered_bits, use_line_collection=True)
plt.title("Восстановленная битовая последовательность")
plt.xlabel("Отсчёт")
plt.ylabel("Значение бита")

plt.tight_layout()
plt.show()
