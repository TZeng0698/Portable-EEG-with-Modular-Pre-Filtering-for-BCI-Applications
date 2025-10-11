import sys
import threading
import csv
from datetime import datetime, timedelta
import numpy as np
import serial
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from scipy.signal import iirnotch, lfilter, butter
from queue import Queue

# Constants for exponential mapping
MIN_SCALE = 1           # µV
MAX_SCALE = 1_000_000   # µV
MIN_TIME = 0.1          # seconds
MAX_TIME = 10.0         # seconds

# Sampling rate configuration
SAMPLE_RATES = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
SAMPLE_RATE_COMMANDS = {
    250: '110',   # DR=0b110
    500: '101',   # DR=0b101
    1000: '100',  # DR=0b100
    2000: '011',  # DR=0b011
    4000: '010',  # DR=0b010
    8000: '001',  # DR=0b001
    16000: '000', # DR=0b000
    32000: '000'  # DR=0b000 (HR mode only)
}

class UnitAxisItem(pg.AxisItem):
    def __init__(self, orientation, **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self.factor = 1.0
        self.setStyle(textFillLimits=[(0, 0.7)], tickFont=QtWidgets.QApplication.font())
    def tickStrings(self, values, scale, spacing):
        strs = []
        for v in values:
            s = v * self.factor
            strs.append(f"{s:.2f}" if abs(s) < 10 else f"{s:.1f}")
        return strs

def choose_unit_and_factor(scale_value):
    if scale_value < 1_000:
        return "µV", 1.0
    elif scale_value < 1_000_000:
        return "mV", 1e-3
    else:
        return "V", 1e-6

class PlotArea(QtWidgets.QWidget):
    pop_requested = QtCore.pyqtSignal(int)
    gain_requested = QtCore.pyqtSignal(int, int)
    short_requested = QtCore.pyqtSignal(int, bool)

    def __init__(self, sample_rate=250, channels=4, buffer_size=500,
                 default_scale=1000, default_time=2.0, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.sample_rate = sample_rate
        self.default_scale = default_scale
        self.default_time = default_time
        self.paused = False
        self.real_time_notch_enabled = False
        self.real_time_hp_enabled = False
        self.notch_freq = 60  # Default notch frequency

        # Per-channel time windows and buffers
        self.channel_times = [default_time] * channels
        self.buffer_sizes = [int(self.sample_rate * default_time)] * channels
        self.time_axes = [np.linspace(-default_time, 0, self.buffer_sizes[i]) for i in range(channels)]

        # NumPy ring buffers
        self.data_buffers = [np.zeros(self.buffer_sizes[i]) for i in range(channels)]
        self.buffer_indices = [0] * channels
        self.fft_buffer_size = int(self.sample_rate * default_time * 2)  # Double FFT buffer for better resolution
        self.fft_buffer = np.zeros(self.fft_buffer_size)
        self.fft_buffer_index = 0
        self.ac_enabled = [False] * channels

        # Single IIR notch filter (Q=30 for 59–61 Hz coverage)
        self.notch_Q = 10
        self.notch_b, self.notch_a = iirnotch(self.notch_freq, self.notch_Q, self.sample_rate)
        self.notch_zi = [np.zeros(len(self.notch_a) - 1) for _ in range(self.channels)]

        # Real-time high-pass filter (1 Hz, 5th order Butterworth)
        self.hp_b, self.hp_a = butter(5, 3 / (self.sample_rate / 2.0), btype='highpass')
        self.hp_zi = [np.zeros(len(self.hp_a) - 1) for _ in range(self.channels)]

        # Precompute FFT frequencies
        self.freqs = np.fft.rfftfreq(self.fft_buffer_size, d=1.0/self.sample_rate)

        # Batch processing for real-time filtering
        self.batch_size = max(1, int(self.sample_rate * 0.05))  # 50 ms batches
        self.batch_buffers = [[] for _ in range(self.channels)]

        # Layout
        layout = QtWidgets.QGridLayout(self)
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        colors = [(255,0,0), (0,150,0), (0,0,255), (255,0,255)]
        self.plots = []
        self.curves = []
        self.vpp_labels = []

        # Create channel plots + controls
        for i in range(channels):
            left_axis = UnitAxisItem('left')
            p = pg.PlotWidget(axisItems={'left': left_axis})
            p.setBackground('w')
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=True, y=True)
            p.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode)
            unit, factor = choose_unit_and_factor(default_scale)
            p.setLabel('left', f'Ch {i+1} Voltage ({unit})', **{'font-weight': 'bold'})
            p.setLabel('bottom', 'Time (s)', **{'font-weight': 'bold'})
            p.getAxis('left').factor = factor
            p.setYRange(-default_scale, default_scale)
            p.setXRange(-default_time, 0)
            curve = p.plot(self.time_axes[i], np.zeros(self.buffer_sizes[i]), pen=pg.mkPen(color=colors[i], width=2))
            
            vpp_label = pg.TextItem(text='Vpp: -- µV', anchor=(1, 0), color=(0, 0, 0))
            vpp_label.setPos(0, default_scale * 0.9)
            p.addItem(vpp_label)
            layout.addWidget(p, i, 0)

            ctrl = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(ctrl)
            vbox.setContentsMargins(0,0,0,0)
            vbox.setSpacing(4)

            btn_pop = QtWidgets.QPushButton('Pop')
            btn_pop.setFixedWidth(40)
            btn_pop.clicked.connect(lambda _, ch=i: self.pop_requested.emit(ch))
            vbox.addWidget(btn_pop)

            chk_ac = QtWidgets.QCheckBox('AC')
            chk_ac.stateChanged.connect(lambda state, ch=i: self.toggle_ac(ch, state))
            vbox.addWidget(chk_ac)

            chk_short = QtWidgets.QCheckBox('Short')
            chk_short.stateChanged.connect(lambda state, ch=i: self.short_requested.emit(ch+1, state == QtCore.Qt.Checked))
            vbox.addWidget(chk_short)

            vbox.addWidget(QtWidgets.QLabel('Scale'))
            sld_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld_scale.setRange(0, 100)
            pos_scale = int((np.log10(default_scale) - np.log10(MIN_SCALE)) /
                           (np.log10(MAX_SCALE) - np.log10(MIN_SCALE)) * 100)
            sld_scale.setValue(pos_scale)
            sld_scale.valueChanged.connect(lambda val, ch=i: self.update_channel_scale_exp(ch, val))
            vbox.addWidget(sld_scale)
            setattr(self, f'scale_slider_{i}', sld_scale)

            vbox.addWidget(QtWidgets.QLabel('Gain'))
            cb_gain = QtWidgets.QComboBox()
            for g in [1,2,4,6,8,12]:
                cb_gain.addItem(f'{g}×', g)
            cb_gain.currentIndexChanged.connect(lambda idx, ch=i, b=cb_gain: self.gain_requested.emit(ch+1, b.currentData()))
            cb_gain.setFixedWidth(60)
            vbox.addWidget(cb_gain)
            setattr(self, f'gain_box_{i}', cb_gain)

            vbox.addWidget(QtWidgets.QLabel('Time'))
            sld_time = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld_time.setRange(0, 100)
            pos_time = int((np.log10(default_time) - np.log10(MIN_TIME)) /
                          (np.log10(MAX_TIME) - np.log10(MIN_TIME)) * 100)
            sld_time.setValue(pos_time)
            sld_time.valueChanged.connect(lambda val, ch=i: self.update_channel_time_exp(ch, val))
            vbox.addWidget(sld_time)
            setattr(self, f'time_slider_{i}', sld_time)

            vbox.addStretch()
            layout.addWidget(ctrl, i, 1)

            self.plots.append(p)
            self.curves.append(curve)
            self.vpp_labels.append(vpp_label)

        p_fft = pg.PlotWidget()
        p_fft.setBackground('w')
        p_fft.showGrid(x=True, y=True, alpha=0.3)
        p_fft.setMenuEnabled(False)
        p_fft.setMouseEnabled(x=True, y=True)
        p_fft.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode)
        p_fft.setLabel('left', 'Amplitude(uV)', **{'font-weight': 'bold'})
        p_fft.setLabel('bottom', 'Frequency (Hz)', **{'font-weight': 'bold'})
        p_fft.setXRange(0.5, 150)
        self.fft_curve = p_fft.plot(self.freqs, np.zeros_like(self.freqs), pen=pg.mkPen(width=2))
        layout.addWidget(p_fft, channels, 0, 1, 2)

        # FFT update timer (every 500 ms)
        self.fft_update_timer = QtCore.QTimer(self)
        self.fft_update_timer.setInterval(500)
        self.fft_update_timer.timeout.connect(self.update_fft)
        self.fft_update_timer.start()

    def update_time_axis(self, channel):
        self.time_axes[channel] = np.linspace(-self.channel_times[channel], 0, self.buffer_sizes[channel])

    def update_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        # Update notch filter for new sample rate
        self.notch_b, self.notch_a = iirnotch(self.notch_freq, self.notch_Q, self.sample_rate)
        self.hp_b, self.hp_a = butter(5, 1.0 / (self.sample_rate / 2.0), btype='highpass')
        self.notch_zi = [np.zeros(len(self.notch_a) - 1) for _ in range(self.channels)]
        self.hp_zi = [np.zeros(len(self.hp_a) - 1) for _ in range(self.channels)]
        self.batch_size = max(1, int(self.sample_rate * 0.05))  # Update batch size
        for i in range(self.channels):
            self.update_buffer_size(i, self.channel_times[i])
            self.plots[i].setXRange(-self.channel_times[i], 0)
            pos_time = int((np.log10(self.channel_times[i]) - np.log10(MIN_TIME)) /
                          (np.log10(MAX_TIME) - np.log10(MIN_TIME)) * 100)
            slider = getattr(self, f'time_slider_{i}')
            slider.blockSignals(True)
            slider.setValue(pos_time)
            slider.blockSignals(False)
            self.batch_buffers[i] = []
        self.fft_buffer_size = int(self.sample_rate * self.default_time * 2)
        self.fft_buffer = np.zeros(self.fft_buffer_size)
        self.fft_buffer_index = 0
        self.freqs = np.fft.rfftfreq(self.fft_buffer_size, d=1.0/self.sample_rate)
        print(f"DEBUG: Updated sample rate to {sample_rate} SPS, FFT buffer_size={self.fft_buffer_size}, batch_size={self.batch_size}")

    def update_buffer_size(self, channel, time_window):
        self.channel_times[channel] = time_window
        new_size = int(self.sample_rate * time_window)
        old_buffer = self.data_buffers[channel]
        self.buffer_sizes[channel] = new_size
        self.data_buffers[channel] = np.zeros(new_size)
        self.buffer_indices[channel] = 0
        if len(old_buffer) > 0:
            self.data_buffers[channel][:min(new_size, len(old_buffer))] = old_buffer[-min(new_size, len(old_buffer)):]
        self.update_time_axis(channel)
        print(f"DEBUG: Ch {channel+1} updated buffer_size={new_size}, time_window={time_window:.2f}s")

    def set_notch_freq(self, freq):
        """Update notch filter frequency and recalculate coefficients."""
        self.notch_freq = freq
        if freq == 0:
            self.real_time_notch_enabled = False
        else:
            self.notch_b, self.notch_a = iirnotch(self.notch_freq, self.notch_Q, self.sample_rate)
            self.notch_zi = [np.zeros(len(self.notch_a) - 1) for _ in range(self.channels)]
        print(f"DEBUG: Notch filter set to {freq} Hz {'(disabled)' if freq == 0 else ''}")

    @QtCore.pyqtSlot(list)
    def buffer_data(self, values):
        if self.paused:
            return
        if self.real_time_hp_enabled or self.real_time_notch_enabled:
            # Accumulate samples in batch buffers
            for i in range(self.channels):
                self.batch_buffers[i].append(values[i])
            # Process batch when enough samples are collected
            if len(self.batch_buffers[0]) >= self.batch_size:
                print(f"DEBUG: Processing batch of {len(self.batch_buffers[0])} samples")
                filtered_data = []
                for i in range(self.channels):
                    if len(self.batch_buffers[i]) == 0:
                        print(f"DEBUG: Warning: batch_buffers[{i}] is empty, skipping filter")
                        filtered_data.append([0] * self.batch_size)
                        continue
                    batch = np.array(self.batch_buffers[i])
                    filtered = batch
                    if self.real_time_hp_enabled:
                        filtered, self.hp_zi[i] = lfilter(self.hp_b, self.hp_a, filtered, zi=self.hp_zi[i])
                    if self.real_time_notch_enabled and self.notch_freq != 0:
                        filtered, self.notch_zi[i] = lfilter(self.notch_b, self.notch_a, filtered, zi=self.notch_zi[i])
                    filtered_data.append(filtered)
                    # Update data buffer
                    for j, val in enumerate(filtered):
                        idx = self.buffer_indices[i] % self.buffer_sizes[i]
                        self.data_buffers[i][idx] = val
                        self.buffer_indices[i] += 1
                # Update FFT buffer with mean of filtered batch
                mean_batch = np.mean(filtered_data, axis=0)
                for val in mean_batch:
                    idx = self.fft_buffer_index % self.fft_buffer_size
                    self.fft_buffer[idx] = val
                    self.fft_buffer_index += 1
                # Clear batch buffers after processing
                for i in range(self.channels):
                    self.batch_buffers[i].clear()
        else:
            # No filtering, update buffers directly
            for i in range(self.channels):
                idx = self.buffer_indices[i] % self.buffer_sizes[i]
                self.data_buffers[i][idx] = values[i]
                self.buffer_indices[i] += 1
            idx = self.fft_buffer_index % self.fft_buffer_size
            self.fft_buffer[idx] = np.mean(values)
            self.fft_buffer_index += 1

    def update_display(self):
        for i, curve in enumerate(self.curves):
            data = self.data_buffers[i].copy()
            if self.ac_enabled[i]:
                data -= np.mean(data)
            curve.setData(self.time_axes[i], data)
            vpp = np.ptp(data)
            unit, factor = choose_unit_and_factor(vpp)
            self.vpp_labels[i].setText(f'Vpp: {vpp * factor:.2f} {unit}')

    def update_fft(self):
        if self.fft_buffer_index == 0:
            return
        buf = self.fft_buffer.copy()
        N = min(self.fft_buffer_index, self.fft_buffer_size)
        if N > 0:
            Y = np.fft.rfft(buf[:N])
            amp = (2.0 / N) * np.abs(Y)
            amp[0] /= 2
            if N % 2 == 0:
                amp[-1] /= 2
            freqs = np.fft.rfftfreq(N, d=1.0/self.sample_rate)
            mask = (freqs >= 0.5) & (freqs <= 150)
            self.fft_curve.setData(freqs[mask], amp[mask])
            # Debug power around 60 Hz
            idx_59_61 = (freqs >= 59) & (freqs <= 61)
            if np.any(idx_59_61):
                power_60hz = np.mean(amp[idx_59_61])
                print(f"DEBUG: Power in 59–61 Hz range: {power_60hz:.2f} µV")

    def apply_notch_filter(self):
        if not self.paused:
            print("DEBUG: Pause notch filter skipped (not paused)")
            return
        if self.notch_freq == 0:
            print("DEBUG: Notch filter disabled")
            return
        print(f"DEBUG: Applying {self.notch_freq} Hz IIR notch filter (Q={self.notch_Q}) at {self.sample_rate} SPS")
        for i in range(self.channels):
            filtered_data = lfilter(self.notch_b, self.notch_a, self.data_buffers[i])
            self.data_buffers[i] = filtered_data
        filtered_mean = np.mean([self.data_buffers[i] for i in range(self.channels)], axis=0)
        self.fft_buffer = filtered_mean[-self.fft_buffer_size:]
        self.update_display()
        self.update_fft()

    def apply_hp_filter(self):
        if not self.paused:
            return
        print(f"DEBUG: Applying 1 Hz high-pass filter at {self.sample_rate} SPS")
        for i in range(self.channels):
            filtered_data = lfilter(self.hp_b, self.hp_a, self.data_buffers[i])
            self.data_buffers[i] = filtered_data
        filtered_mean = np.mean([self.data_buffers[i] for i in range(self.channels)], axis=0)
        self.fft_buffer = filtered_mean[-self.fft_buffer_size:]
        self.update_display()
        self.update_fft()

    def toggle_pause(self, state):
        self.paused = state == QtCore.Qt.Checked
        if not self.paused:
            for i in range(self.channels):
                self.data_buffers[i] = np.zeros(self.buffer_sizes[i])
                self.buffer_indices[i] = 0
                self.batch_buffers[i].clear()
            self.fft_buffer = np.zeros(self.fft_buffer_size)
            self.fft_buffer_index = 0
            print("DEBUG: Buffers reset on resume")

    def toggle_real_time_notch(self, state):
        self.real_time_notch_enabled = state == QtCore.Qt.Checked
        if self.real_time_notch_enabled:
            self.notch_zi = [np.zeros(len(self.notch_a) - 1) for _ in range(self.channels)]
        for i in range(self.channels):
            self.batch_buffers[i].clear()
        print(f"DEBUG: Real-time {self.notch_freq} Hz IIR notch filter {'enabled' if self.real_time_notch_enabled else 'disabled'}")

    def toggle_real_time_hp(self, state):
        self.real_time_hp_enabled = state == QtCore.Qt.Checked
        if self.real_time_hp_enabled:
            self.hp_zi = [np.zeros(len(self.hp_a) - 1) for _ in range(self.channels)]
        for i in range(self.channels):
            self.batch_buffers[i].clear()
        print(f"DEBUG: Real-time 1 Hz high-pass filter {'enabled' if self.real_time_hp_enabled else 'disabled'}")

    def update_channel_scale_exp(self, ch, slider_val):
        scale = MIN_SCALE * (MAX_SCALE/MIN_SCALE)**(slider_val/100)
        p = self.plots[ch]
        p.setYRange(-scale, scale, padding=0)
        unit, factor = choose_unit_and_factor(scale)
        p.setLabel('left', f'Ch {ch+1} Voltage ({unit})', **{'font-weight': 'bold'})
        p.getAxis('left').factor = factor
        self.vpp_labels[ch].setPos(0, scale * 0.9)

    def update_channel_time_exp(self, ch, slider_val):
        t = MIN_TIME * (MAX_TIME/MIN_TIME)**(slider_val/100)
        self.update_buffer_size(ch, t)
        self.plots[ch].setXRange(-t, 0)
        self.curves[ch].setData(self.time_axes[ch], self.data_buffers[ch])

    def toggle_ac(self, ch, state):
        self.ac_enabled[ch] = (state == QtCore.Qt.Checked)

class PopChannelWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal(int)

    def __init__(self, ch_index, sample_rate=250, buffer_size=500, voltage_scale=1000):
        super().__init__()
        self.ch_index = ch_index
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.voltage_scale = voltage_scale
        self.time_window = buffer_size / sample_rate
        self.setWindowTitle(f"Popped-Out Ch {ch_index+1}")

        central = QtWidgets.QWidget()
        central.setStyleSheet("background-color: white;")
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        left_axis = UnitAxisItem('left')
        self.plot = pg.PlotWidget(axisItems={'left': left_axis})
        self.plot.setBackground('w')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', f'Ch {ch_index+1} Voltage (µV)', **{'font-weight': 'bold'})
        self.plot.setLabel('bottom', 'Time (s)', **{'font-weight': 'bold'})
        self.plot.getAxis('left').factor = 1.0
        self.plot.setYRange(-voltage_scale, voltage_scale)
        self.plot.setXRange(-self.time_window, 0)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode)
        pen = pg.mkPen(color=(255,0,0), width=2)
        t_axis = np.linspace(-buffer_size/sample_rate, 0, buffer_size)
        self.curve = self.plot.plot(t_axis, np.zeros(buffer_size), pen=pen)
        self.vpp_label = pg.TextItem(text='Vpp: -- µV', anchor=(1, 0), color=(0, 0, 0))
        self.vpp_label.setPos(0, voltage_scale * 0.9)
        self.plot.addItem(self.vpp_label)
        layout.addWidget(self.plot)

        controls = QtWidgets.QHBoxLayout()
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        controls.addWidget(self.btn_pause)

        self.sld_vs = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_vs.setRange(0, 100)
        init_vs = int((np.log10(voltage_scale)-np.log10(MIN_SCALE)) / (np.log10(MAX_SCALE)-np.log10(MIN_SCALE)) * 100)
        self.sld_vs.setValue(init_vs)
        self.sld_vs.valueChanged.connect(self.update_voltage_scale)
        controls.addWidget(self.sld_vs)

        self.sld_time = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_time.setRange(0, 100)
        init_time = int((np.log10(self.time_window)-np.log10(MIN_TIME)) / (np.log10(MAX_TIME)-np.log10(MIN_TIME)) * 100)
        self.sld_time.setValue(init_time)
        self.sld_time.valueChanged.connect(self.update_time_window)
        controls.addWidget(self.sld_time)

        layout.addLayout(controls)

        self.buffer = np.zeros(buffer_size)
        self.buffer_index = 0
        self.paused = False
        self.time_axis = t_axis

    def update_data(self, values):
        if self.paused:
            return
        idx = self.buffer_index % self.buffer_size
        self.buffer[idx] = values[self.ch_index]
        self.buffer_index += 1
        data = self.buffer.copy()
        self.curve.setData(self.time_axis, data)
        vpp = np.ptp(data)
        unit, factor = choose_unit_and_factor(vpp)
        self.vpp_label.setText(f'Vpp: {vpp * factor:.2f} {unit}')

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("Resume" if self.paused else "Pause")

    def update_voltage_scale(self, slider_val):
        vs = MIN_SCALE * (MAX_SCALE/MIN_SCALE)**(slider_val/100)
        self.voltage_scale = vs
        unit, factor = choose_unit_and_factor(vs)
        self.plot.setLabel('left', f'Ch {self.ch_index+1} Voltage ({unit})', **{'font-weight': 'bold'})
        self.plot.getAxis('left').factor = factor
        self.plot.setYRange(-vs, vs, padding=0)
        self.vpp_label.setPos(0, vs * 0.9)

    def update_time_window(self, slider_val):
        t = MIN_TIME * (MAX_TIME/MIN_TIME)**(slider_val/100)
        self.time_window = t
        self.buffer_size = int(self.sample_rate * t)
        self.time_axis = np.linspace(-self.time_window, 0, self.buffer_size)
        old_buffer = self.buffer
        self.buffer = np.zeros(self.buffer_size)
        self.buffer[:min(self.buffer_size, len(old_buffer))] = old_buffer[-min(self.buffer_size, len(old_buffer)):]
        self.buffer_index = min(self.buffer_index, self.buffer_size)
        self.plot.setXRange(-t, 0)
        self.curve.setData(self.time_axis, self.buffer)
        print(f"DEBUG: Pop Ch {self.ch_index+1} time window updated to {t:.2f}s")

    def update_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        self.buffer_size = int(self.sample_rate * self.time_window)
        self.time_axis = np.linspace(-self.time_window, 0, self.buffer_size)
        old_buffer = self.buffer
        self.buffer = np.zeros(self.buffer_size)
        self.buffer[:min(self.buffer_size, len(old_buffer))] = old_buffer[-min(self.buffer_size, len(old_buffer)):]
        self.buffer_index = min(self.buffer_index, self.buffer_size)
        self.plot.setXRange(-self.time_window, 0)
        self.curve.setData(self.time_axis, self.buffer)
        pos_time = int((np.log10(self.time_window) - np.log10(MIN_TIME)) /
                      (np.log10(MAX_TIME) - np.log10(MIN_TIME)) * 100)
        self.sld_time.blockSignals(True)
        self.sld_time.setValue(pos_time)
        self.sld_time.blockSignals(False)
        print(f"DEBUG: Pop Ch {self.ch_index+1} updated sample rate to {sample_rate} SPS")

    def closeEvent(self, event):
        self.closed.emit(self.ch_index)
        super().closeEvent(event)

class MainWindow(QtWidgets.QMainWindow):
    data_received = QtCore.pyqtSignal(list)
    raw_line_received = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG-ADS1298 Viewer + FFT + Controls")
        central = QtWidgets.QWidget()
        central.setStyleSheet("background-color: white;")
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.serial_port = None
        self.running = False
        self._manually_disconnected = False
        self.sample_rate = 250
        self.mode = 'LP'
        self.time_interval = 2.0
        self.recording = False
        self.record_file = None
        self.csv_writer = None
        self.last_flush_time = None
        self.record_start_time = None
        self.record_sample_count = 0

        self.plot_area = PlotArea(sample_rate=self.sample_rate,
                                 channels=4,
                                 buffer_size=int(self.sample_rate * self.time_interval),
                                 default_scale=1000,
                                 default_time=self.time_interval)
        layout.addWidget(self.plot_area, stretch=5)
        self.data_received.connect(self.plot_area.buffer_data)
        self.data_received.connect(self.record_data)
        self.plot_area.pop_requested.connect(self.pop_channel)
        self.plot_area.gain_requested.connect(self.send_gain)
        self.plot_area.short_requested.connect(self.send_short)

        ctrl = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl, stretch=1)

        ctrl.addWidget(QtWidgets.QLabel('Serial Port:'))
        self.port_input = QtWidgets.QLineEdit("COM7")
        ctrl.addWidget(self.port_input)

        ctrl.addWidget(QtWidgets.QLabel('Baud Rate:'))
        self.baud_input = QtWidgets.QLineEdit('115200')
        ctrl.addWidget(self.baud_input)

        conn_h = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton('Connect')
        self.btn_connect.clicked.connect(self.connect_serial)
        conn_h.addWidget(self.btn_connect)
        self.btn_disconnect = QtWidgets.QPushButton('Disconnect')
        self.btn_disconnect.clicked.connect(self.disconnect_serial)
        self.btn_disconnect.setEnabled(False)
        conn_h.addWidget(self.btn_disconnect)
        ctrl.addLayout(conn_h)

        ctrl.addWidget(QtWidgets.QLabel('Mode:'))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(['LP', 'HR'])
        self.mode_combo.setCurrentText('LP')
        self.mode_combo.currentTextChanged.connect(self.update_mode)
        ctrl.addWidget(self.mode_combo)

        ctrl.addWidget(QtWidgets.QLabel('Sample Rate:'))
        self.sr_combo = QtWidgets.QComboBox()
        for sr in SAMPLE_RATES:
            self.sr_combo.addItem(f'{sr} SPS', sr)
        self.sr_combo.setCurrentText('250 SPS')
        self.sr_combo.currentIndexChanged.connect(self.update_sample_rate)
        self.update_sample_rate_combo()
        ctrl.addWidget(self.sr_combo)

        ctrl.addWidget(QtWidgets.QLabel('Notch Filter:'))
        self.notch_combo = QtWidgets.QComboBox()
        self.notch_combo.addItems(['60 Hz', '50 Hz', 'Off'])
        self.notch_combo.setCurrentText('60 Hz')
        self.notch_combo.currentTextChanged.connect(self.update_notch_freq)
        ctrl.addWidget(self.notch_combo)

        self.chk_pause = QtWidgets.QCheckBox('Pause Display')
        self.chk_pause.stateChanged.connect(self.plot_area.toggle_pause)
        self.chk_pause.stateChanged.connect(lambda state: print(f"DEBUG: Pause state changed to {state == QtCore.Qt.Checked}"))
        ctrl.addWidget(self.chk_pause)

        self.chk_real_time_notch = QtWidgets.QCheckBox('Real-Time Notch')
        self.chk_real_time_notch.stateChanged.connect(self.plot_area.toggle_real_time_notch)
        ctrl.addWidget(self.chk_real_time_notch)

        self.chk_real_time_hp = QtWidgets.QCheckBox('Real-Time HP 1Hz')
        self.chk_real_time_hp.stateChanged.connect(self.plot_area.toggle_real_time_hp)
        ctrl.addWidget(self.chk_real_time_hp)

        btn_notch = QtWidgets.QPushButton('Apply Notch Filter')
        btn_notch.clicked.connect(self.plot_area.apply_notch_filter)
        btn_notch.setEnabled(False)
        self.chk_pause.stateChanged.connect(lambda state: btn_notch.setEnabled(state == QtCore.Qt.Checked))
        ctrl.addWidget(btn_notch)

        btn_hp = QtWidgets.QPushButton('Apply HP Filter')
        btn_hp.clicked.connect(self.plot_area.apply_hp_filter)
        btn_hp.setEnabled(False)
        self.chk_pause.stateChanged.connect(lambda state: btn_hp.setEnabled(state == QtCore.Qt.Checked))
        ctrl.addWidget(btn_hp)

        btn_get_sr = QtWidgets.QPushButton('GET_SR')
        btn_get_sr.clicked.connect(self.query_sample_rate)
        ctrl.addWidget(btn_get_sr)
        self.sr_display = QtWidgets.QLabel('Device SR: -- Hz')
        ctrl.addWidget(self.sr_display)
        self.sr_meas = QtWidgets.QLabel('Measured SR: -- Hz')
        ctrl.addWidget(self.sr_meas)

        self.sample_counter = 0
        sr_timer = QtCore.QTimer(self)
        sr_timer.timeout.connect(self._update_measured_sr)
        sr_timer.start(1000)

        self.btn_record = QtWidgets.QPushButton('Start Recording')
        self.btn_record.clicked.connect(self.toggle_recording)
        ctrl.addWidget(self.btn_record)

        ctrl.addWidget(QtWidgets.QLabel('Send Command:'))
        cmd_h = QtWidgets.QHBoxLayout()
        self.cmd_input = QtWidgets.QLineEdit()
        cmd_h.addWidget(self.cmd_input)
        btn_send = QtWidgets.QPushButton('Send')
        btn_send.clicked.connect(self.send_command)
        cmd_h.addWidget(btn_send)
        ctrl.addLayout(cmd_h)

        ctrl.addWidget(QtWidgets.QLabel('Serial Log (latest every 10s):'))
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.log_text.setFixedHeight(100)
        ctrl.addWidget(self.log_text)

        ctrl.addStretch()

        # Plot timer at 20 FPS (50 ms)
        plot_timer = QtCore.QTimer(self)
        plot_timer.setInterval(50)
        plot_timer.timeout.connect(self.plot_area.update_display)
        plot_timer.start()

        self.thread_timer = QtCore.QTimer(self)
        self.thread_timer.setInterval(max(1, int(1000/self.sample_rate)))
        self.thread_timer.start()

        self.raw_line_received.connect(self._on_new_log_line)

        self.pop_windows = [None] * 4

    def update_notch_freq(self, freq_text):
        """Update the notch filter frequency based on combo box selection."""
        if freq_text == 'Off':
            freq = 0
        else:
            freq = int(freq_text.split()[0])
        self.plot_area.set_notch_freq(freq)

    def update_mode(self, mode):
        self.mode = mode
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(f'{mode}\n'.encode())
            self.serial_port.write(b'START\n')
        self.update_sample_rate_combo()
        if self.sample_rate == 32000 and mode == 'LP':
            self.sr_combo.setCurrentText('250 SPS')
            self.update_sample_rate()

    def update_sample_rate_combo(self):
        self.sr_combo.blockSignals(True)
        current_sr = self.sr_combo.currentText()
        self.sr_combo.clear()
        valid_rates = [sr for sr in SAMPLE_RATES if sr != 32000 or self.mode == 'HR']
        for sr in valid_rates:
            self.sr_combo.addItem(f'{sr} SPS', sr)
        if current_sr in [f'{sr} SPS' for sr in valid_rates]:
            self.sr_combo.setCurrentText(current_sr)
        else:
            self.sr_combo.setCurrentText('250 SPS')
        self.sr_combo.blockSignals(False)

    def update_sample_rate(self):
        sr = self.sr_combo.currentData()
        if sr != self.sample_rate:
            self.sample_rate = sr
            self.plot_area.update_sample_rate(self.sample_rate)
            self.thread_timer.setInterval(max(1, int(1000/self.sample_rate)))
            if self.serial_port and self.serial_port.is_open:
                cmd = f'SAMPLERATE {sr}\n'
                self.serial_port.write(cmd.encode())
                self.serial_port.write(b'START\n')
            for win in self.pop_windows:
                if win is not None and win.isVisible():
                    win.update_sample_rate(self.sample_rate)
            if self.recording:
                self.record_start_time = datetime.now()
                self.record_sample_count = 0
            print(f"DEBUG: MainWindow updated sample rate to {sr} SPS")

    def connect_serial(self):
        port = self.port_input.text().strip()
        baud_text = self.baud_input.text().strip()
        if not port or not baud_text:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Enter valid port & baud.')
            return
        try:
            baud = int(baud_text)
            self.serial_port = serial.Serial(port, baud, timeout=0.1)
            self.running = True
            self._manually_disconnected = False
            reader = threading.Thread(target=self._serial_reader, daemon=True)
            reader.start()
            self.serial_port.write(f'{self.mode}\n'.encode())
            self.serial_port.write(f'SAMPLERATE {self.sample_rate}\n'.encode())
            self.serial_port.write(b'START\n')
            QtWidgets.QMessageBox.information(self, 'Connected', f'{port}@{baud}')
            self.port_input.setEnabled(False)
            self.baud_input.setEnabled(False)
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.mode_combo.setEnabled(False)
            self.sr_combo.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Cannot open port:\n{e}')

    def disconnect_serial(self):
        self._manually_disconnected = True
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(b'STOP\n')
            except:
                pass
            self.serial_port.close()
        self.serial_port = None
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.setEnabled(True)
        self.port_input.setEnabled(True)
        self.baud_input.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.sr_combo.setEnabled(True)

    def _serial_reader(self):
        HEADER = bytes([0xAA, 0x55])
        FOOTER = bytes([0x0D, 0x0A])
        PACKET_SIZE = 16  # 2 header + 12 data + 2 footer

        buffer = bytearray()

        while self.running:
            try:
                buffer.extend(self.serial_port.read(PACKET_SIZE - len(buffer)))
                if len(buffer) < PACKET_SIZE:
                    continue

                header_index = buffer.find(HEADER)
                if header_index == -1:
                    buffer.clear()
                    continue

                if len(buffer) - header_index < PACKET_SIZE:
                    buffer[:] = buffer[header_index:]
                    continue

                packet = buffer[header_index:header_index + PACKET_SIZE]
                del buffer[:header_index + PACKET_SIZE]

                if packet[-2:] != FOOTER:
                    print("Invalid packet footer")
                    continue

                vals = []
                for ch in range(4):
                    base = 2 + ch * 3
                    b0, b1, b2 = packet[base:base+3]
                    value = (b0 << 16) | (b1 << 8) | b2
                    if value & 0x800000:  # sign extend
                        value -= 1 << 24
                    value = value/2.08  # Convert to mV
                    vals.append(value)

                self.sample_counter += 1
                self.data_received.emit(vals)

            except Exception as e:
                print(f"Serial read error: {e}")
                if not self._manually_disconnected:
                    self.running = False
                    QtWidgets.QApplication.postEvent(self, QtCore.QEvent(QtCore.QEvent.User))
                break

    def customEvent(self, event):
        if event.type() == QtCore.QEvent.User:
            self.disconnect_serial()
            QtWidgets.QMessageBox.critical(self, 'Error', 'Serial connection lost.')

    def _on_new_log_line(self, line):
        now = datetime.now()
        if not hasattr(self, '_last_log_time') or (now - self._last_log_time).total_seconds() >= 10:
            self.log_text.setPlainText(f'[{now.isoformat()}] {line}')
            self._last_log_time = now

    @QtCore.pyqtSlot(list)
    def record_data(self, values):
        if self.recording and self.csv_writer:
            timestamp = self.record_start_time + timedelta(seconds=self.record_sample_count / self.sample_rate)
            self.csv_writer.writerow([timestamp.isoformat()] + values[:4])
            self.record_sample_count += 1
            now = datetime.now()
            if not hasattr(self, '_last_flush_time') or (now - self._last_flush_time).total_seconds() >= 1:
                self.record_file.flush()
                self._last_flush_time = now
                print(f"DEBUG: Recorded sample {self.record_sample_count}, timestamp={timestamp.isoformat()}")

    def _update_measured_sr(self):
        self.sr_meas.setText(f'Measured SR: {self.sample_counter} Hz')
        self.sample_counter = 0

    def query_sample_rate(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(b'GET_SR\n')

    def send_gain(self, channel, gain):
        if self.serial_port and self.serial_port.is_open:
            cmd = f'GAIN {channel} {gain}\n'
            try:
                self.serial_port.write(cmd.encode())
            except:
                pass

    def send_short(self, channel, state):
        if self.serial_port and self.serial_port.is_open:
            cmd = f'{"SHORT" if state else "NORMAL"} {channel}\n'
            try:
                self.serial_port.write(cmd.encode())
            except:
                pass

    def send_command(self):
        cmd = self.cmd_input.text().strip()
        if cmd and self.serial_port and self.serial_port.is_open:
            self.serial_port.write((cmd + '\n').encode())
            self.cmd_input.clear()

    def toggle_recording(self):
        if not self.recording:
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Recording', '', 'CSV Files (*.csv)')
            if fname:
                try:
                    self.record_file = open(fname, 'w', newline='')
                    self.csv_writer = csv.writer(self.record_file)
                    self.csv_writer.writerow(['timestamp', 'ch1', 'ch2', 'ch3', 'ch4'])
                    self.recording = True
                    self.btn_record.setText('Stop Recording')
                    self._last_flush_time = datetime.now()
                    self.record_start_time = datetime.now()
                    self.record_sample_count = 0
                    print("DEBUG: Started recording to", fname)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Error', f'Cannot open file:\n{e}')
        else:
            self.recording = False
            self.btn_record.setText('Start Recording')
            if self.record_file:
                self.record_file.close()
                self.record_file = None
                self.csv_writer = None
                print("DEBUG: Stopped recording")

    def pop_channel(self, ch):
        if self.pop_windows[ch] is None:
            slider_val = getattr(self.plot_area, f'scale_slider_{ch}').value()
            scale = MIN_SCALE * (MAX_SCALE/MIN_SCALE)**(slider_val/100)
            time_window = self.plot_area.channel_times[ch]
            win = PopChannelWindow(ch, sample_rate=self.sample_rate,
                                  buffer_size=int(self.sample_rate * time_window),
                                  voltage_scale=scale)
            win.closed.connect(lambda idx, w=win: self._close_pop(idx, w))
            self.data_received.connect(win.update_data)
            self.pop_windows[ch] = win
            win.show()
        else:
            w = self.pop_windows[ch]
            if not w.isVisible():
                w.show()
            w.raise_()
            w.activateWindow()

    def _close_pop(self, idx, win):
        self.pop_windows[idx] = None

    def closeEvent(self, event):
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(b'STOP\n')
            except:
                pass
            self.serial_port.close()
        if self.record_file:
            self.record_file.close()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())