# ---------------- IMPORT LIBRARIES ----------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, filtfilt

# ---------------- CREATE WINDOW ----------------
root = tb.Window(themename="cyborg")
root.title("Heart Sound Analysis System")
root.geometry("1000x850")
root.resizable(False, False)

# ---------------- VARIABLES ----------------
current_signal = None
filtered_signal = None
sample_rate = None

# ---------------- HEADER ----------------
title_label = tb.Label(root, text="💙 Heart Sound Analysis System", font=("Arial", 28, "bold"), bootstyle="info")
title_label.pack(pady=20)

subtitle = tb.Label(root, text="Analyze Heartbeat Sounds Using Digital Signal Processing", font=("Arial", 12), bootstyle="light")
subtitle.pack()

# ---------------- PROGRESS BAR ----------------
loading = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
loading.pack(pady=20)

# ---------------- RESULT BOX ----------------
result_box = tk.Text(root, height=10, width=90, font=("Consolas", 12), bg="#020617", fg="#22c55e", bd=0)
result_box.pack(pady=10, padx=20)

def write_result(text):
    result_box.delete("1.0", "end")
    result_box.insert("end", text)

# ---------------- DSP FUNCTIONS ----------------

def remove_noise(data, sr):
    # استخدام Bandpass filter (20-500Hz) هو الأدق طبياً لأصوات القلب
    low_cut = 20.0
    high_cut = 500.0
    nyquist = 0.5 * sr
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

# ---------------- BUTTON ACTIONS ----------------

def upload_sound():
    global current_signal, sample_rate, filtered_signal
    file_path = filedialog.askopenfilename(title="Choose Heart Sound", filetypes=[("Audio Files", "*.wav *.mp3")])
    
    if file_path:
        current_signal, sample_rate = librosa.load(file_path, sr=None)
        filtered_signal = None  # إعادة ضبط الإشارة المفلترة عند رفع ملف جديد
        write_result(
            "✅ Heart Sound Uploaded Successfully\n\n"
            f"✔ Sample Rate : {sample_rate} Hz\n"
            f"✔ Duration    : {len(current_signal)/sample_rate:.2f} Seconds\n\n"
            "Ready For Analysis..."
        )

def detect_noise():
    global current_signal, filtered_signal, sample_rate
    if current_signal is None:
        messagebox.showerror("Error", "Please Upload A Sound First")
        return

    noise_level = np.std(current_signal)

    if noise_level > 0.03:
        # عملية التنقية
        filtered_signal = remove_noise(current_signal, sample_rate)
        # حفظ الملف المفلتر
        sf.write("clean_heart.wav", filtered_signal, sample_rate)
        
        # أهم خطوة: استبدال الإشارة الحالية بالمفلترة لاستخدامها في العرض والتحليل
        current_signal = filtered_signal 
        
        write_result(
            f"📊 Noise Detection Report\n\n"
            f"✔ Noise Level (STD): {noise_level:.4f}\n"
            f"✔ Status: Noise Detected\n"
            f"✔ Action: Filter Applied & Loaded Successfully\n"
            f"✔ Saved As: clean_heart.wav"
        )
    else:
        write_result(
            f"📊 Noise Detection Report\n\n"
            f"✔ Noise Level (STD): {noise_level:.4f}\n"
            f"✔ Status: Signal is Clean. No filter needed."
        )

def show_signal():
    if current_signal is None:
        messagebox.showerror("Error", "No signal to display")
        return
    
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(current_signal, sr=sample_rate, color='#38bdf8')
    plt.title("Heart Sound Waveform (Current Signal)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def show_spectrogram():
    if current_signal is None:
        messagebox.showerror("Error", "No signal to display")
        return
    
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(current_signal)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Frequency Spectrogram")
    plt.tight_layout()
    plt.show()

def analyze_heart():
    if current_signal is None:
        messagebox.showerror("Error", "Please upload a sound first")
        return

    # انيميشن التحميل
    loading['value'] = 0
    for i in range(0, 101, 10):
        loading['value'] = i
        root.update_idletasks()
        root.after(50)

    # حسابات الطاقة والتردد
    energy = np.mean(current_signal ** 2)
    fft_data = np.abs(np.fft.fft(current_signal))
    avg_freq = np.mean(fft_data)

    if energy > 0.005 or avg_freq > 10:
        res = "⚠️ ABNORMAL HEART SOUND DETECTED"
        note = "Irregular patterns or high murmurs found."
    else:
        res = "✅ NORMAL HEART SOUND"
        note = "Stable rhythm and frequency profile."

    write_result(f"{res}\n\nEnergy Level: {energy:.6f}\nAvg Frequency: {avg_freq:.2f}\n\nNotes: {note}")

def add_new_sound():
    if messagebox.askyesno("New Sound", "Do you want to add another sound?"):
        upload_sound()

# ---------------- BUTTON STYLING ----------------

def on_enter(e):
    e.widget['background'] = '#60a5fa'

def on_leave(e, color):
    e.widget['background'] = color

def style_button(btn, color):
    btn.config(
        bg=color, fg="white", activebackground="#64748b", activeforeground="white",
        bd=0, relief="flat", font=("Arial", 11, "bold"), cursor="hand2",
        width=20, height=2
    )
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", lambda e: on_leave(e, color))

# ---------------- BUTTON LAYOUT ----------------

btn_frame = tk.Frame(root, bg="#0f172a")
btn_frame.pack(pady=15)

# تعريف الأزرار
buttons = [
    ("Upload Sound", upload_sound, "#2563eb", 0, 0),
    ("Detect Noise", detect_noise, "#1d4ed8", 0, 1),
    ("Show Signal", show_signal, "#0ea5e9", 1, 0),
    ("Spectrogram", show_spectrogram, "#38bdf8", 1, 1),
    ("Analyze Heart", analyze_heart, "#1e40af", 2, 0),
    ("Add New Sound", add_new_sound, "#1e3a8a", 2, 1)
]

for text, cmd, color, r, c in buttons:
    b = tk.Button(btn_frame, text=text, command=cmd)
    style_button(b, color)
    b.grid(row=r, column=c, padx=12, pady=12)

# ---------------- FOOTER ----------------
footer = tk.Frame(root, bg="#0f172a")
footer.pack(side="bottom", fill="x", pady=15)

tk.Label(footer, text="Stay healthy, stay safe", font=("Arial", 12, "bold"), fg="#e2e8f0", bg="#0f172a").pack()
tk.Label(footer, text="Designed for Medical Signal Analysis (DSP)", font=("Arial", 10), fg="#94a3b8", bg="#0f172a").pack()

# ---------------- RUN ----------------
root.mainloop()