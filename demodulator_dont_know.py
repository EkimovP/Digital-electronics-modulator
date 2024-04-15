import numpy as np
import matplotlib.pyplot as plt


class CostasLoopDemodulator:
    def __init__(self, carrier_frequency, symbol_rate, sampling_rate):
        # Несущая частота
        self.fc = carrier_frequency
        # Скорость символов
        self.rs = symbol_rate
        # Частота дискретизации
        self.fs = sampling_rate
        # Начальное значение оценки фазы
        self.phase_estimate = 0.0

    # Принимает входной сигнал и возвращает демодулированную последовательность битов
    def demodulate(self, signal):
        t = np.arange(len(signal)) / self.fs

        # Генерируем сигнал для демодуляции
        cos_signal = np.cos(2 * np.pi * self.fc * t + self.phase_estimate)
        sin_signal = np.sin(2 * np.pi * self.fc * t + self.phase_estimate)

        # Демодуляция
        # Ин-фазный сигнал
        I = signal * cos_signal
        # Квадратурный сигнал
        Q = signal * sin_signal

        # Оценка фазовой ошибки
        phase_error = np.arctan2(np.sum(Q), np.sum(I))

        # Обновляем оценку фазы
        self.phase_estimate += phase_error

        # Ограничиваем фазу для предотвращения переполнения (от 0 до 2pi)
        self.phase_estimate = np.mod(self.phase_estimate, 2 * np.pi)

        # Декодируем биты (BPSK демодуляция)
        # Если I >= 0, то бит равен 0, иначе 1
        demodulated_bits = np.where(I >= 0, 0, 1)

        return demodulated_bits


# Параметры сигнала и демодулятора
carrier_freq = 500  # частота несущей в Гц
symbol_rate = 10  # скорость символов в символах/сек
sampling_rate = 500  # частота дискретизации в Гц

# Создаем демодулятор
demodulator = CostasLoopDemodulator(carrier_freq, symbol_rate, sampling_rate)

# Генерируем и модулируем сигнал BPSK с шумом и эффектом Доплера
t = np.linspace(0, 1, sampling_rate, endpoint=False)
num_symbols = int(len(t) * symbol_rate / sampling_rate)  # количество символов
# генерируется случайная последовательность символов BPSK
symbol_sequence = np.random.randint(0, 2, num_symbols)
# повторяем каждый символ в с/о с частотой дискретизации (для формирования сигнала)
bpsk_signal = np.repeat(symbol_sequence, sampling_rate // symbol_rate)
# Генерируется несущий сигнал с заданной частотой
carrier_signal = np.cos(2 * np.pi * carrier_freq * t)
modulated_signal = (
    np.sqrt(2) * bpsk_signal * carrier_signal
)  # модулированный сигнал BPSK
noisy_signal = modulated_signal + 0.5 * np.random.randn(len(t))  # добавляем шум

# Демодулируем зашумленный сигнал
demodulated_bits = demodulator.demodulate(noisy_signal)

# Выводим результаты
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, modulated_signal)
plt.title("Модулированный BPSK сигнал")

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal)
plt.title("Зашумленный BPSK сигнал")

plt.subplot(3, 1, 3)
plt.stem(demodulated_bits, markerfmt="ro", linefmt="grey", basefmt=" ")
plt.title("Демодулированная последовательность")

print("Демодулированная последовательность битов: ")
print(demodulated_bits)
print(len(demodulated_bits))

plt.tight_layout()
plt.show()
