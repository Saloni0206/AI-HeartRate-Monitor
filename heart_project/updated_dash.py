import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import pyttsx3
import winsound

# ------------------------------
# Load dataset
# ------------------------------
data = pd.read_csv("heart_disease_uci.csv")

normal_data = data[data['num'] == 0]['trestbps'].tolist()
mild_data = data[data['num'] == 1]['trestbps'].tolist()
moderate_data = data[data['num'] == 2]['trestbps'].tolist()
severe_data = data[data['num'] >= 3]['trestbps'].tolist()

categories = [
    ("Normal (num=0)", normal_data, "#4CAF50"),
    ("Mild (num=1)", mild_data, "#FFC107"),
    ("Moderate (num=2)", moderate_data, "#FF9800"),
    ("Severe (num=3/4)", severe_data, "#F44336")
]

# ------------------------------
# Load model
# ------------------------------
class DummyModel:
    def predict(self, X):
        hr = X[0][0]
        if hr < 60 or hr > 130:
            return [2]
        elif 90 <= hr <= 120:
            return [1]
        else:
            return [0]

try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
        model_loaded = True
        print("✅ Real AI model loaded successfully!")
except Exception as e:
    print("⚠️ Using dummy model:", e)
    model = DummyModel()
    model_loaded = False

# ------------------------------
# Tkinter UI
# ------------------------------
root = tk.Tk()
root.title("💓 AI-Based Live Heart Rate Monitoring Dashboard")
root.geometry("1200x700")
root.config(bg="#f5fff5")

# Header
tk.Label(root, text="🧠 AI-Powered Real-Time Heart Monitoring System",
         font=("Arial", 18, "bold"), bg="#f5fff5", fg="#333").pack(pady=10)

model_status = tk.Label(root, text=f"Model: {'AI Model Loaded ✅' if model_loaded else 'Dummy Model (Testing) ⚙️'}",
                        font=("Arial", 12, "italic"), bg="#f5fff5", fg="#007700")
model_status.pack(pady=5)

status_label = tk.Label(root, text="Initializing...", font=("Arial", 14, "bold"), bg="#f5fff5")
status_label.pack(pady=5)

hr_var = tk.StringVar(value="Heart Rate: -- BPM")
tk.Label(root, textvariable=hr_var, font=("Arial", 15, "bold"), bg="#f5fff5").pack(pady=5)

# ------------------------------
# Voice setup
# ------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

def speak(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()

# ------------------------------
# Layout frames
# ------------------------------
frame_main = tk.Frame(root, bg="#f5fff5")
frame_main.pack(fill="both", expand=True, padx=10, pady=10)

frame_graph = tk.Frame(frame_main, bg="#f5fff5")
frame_graph.pack(side="left", fill="both", expand=True, padx=10, pady=10)

frame_log = tk.Frame(frame_main, bg="#ffffff", relief="solid", bd=1)
frame_log.pack(side="right", fill="y", padx=10, pady=10)

tk.Label(frame_log, text="🩸 Live Alert Logs", font=("Arial", 14, "bold"), bg="#ffffff", fg="#333").pack(pady=5)
log_box = scrolledtext.ScrolledText(frame_log, width=40, height=25, font=("Consolas", 11), bg="#f9f9f9")
log_box.pack(fill="both", expand=True, padx=5, pady=5)
log_box.config(state="disabled")

def log_message(msg):
    log_box.config(state="normal")
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    log_box.config(state="disabled")

# ------------------------------
# Graph setup (4 subplots)
# ------------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 5))
fig.suptitle("Real-Time Heart Rate (Dataset Categories)", fontsize=14)
canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas.get_tk_widget().pack(pady=10)

lines = []
for i, (title, values, color) in enumerate(categories):
    ax = axs[i // 2][i % 2]
    ax.set_title(title)
    ax.set_xlim(0, len(values))
    ax.set_ylim(50, 200)
    ax.grid(True)
    line, = ax.plot([], [], color=color, linewidth=2)
    lines.append((line, values, ax, color))

# ------------------------------
# Live update (smart speech + log)
# ------------------------------
index = 0
last_status = None

def update_graphs():
    global index, last_status
    if index >= max(len(c[1]) for c in categories):
        index = 0

    for line, values, ax, color in lines:
        if index < len(values):
            hr = int(values[index])
            y_data = values[:index + 1]
            x_data = list(range(len(y_data)))
            line.set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view()

            hr_var.set(f"Current Heart Rate: {hr} BPM")
            pred = model.predict([[hr]])[0]
            current_status = None

            if pred == 2 or hr > 160 or hr < 60:
                current_status = "critical"
                status_label.config(text=f"🚨 CRITICAL ALERT: {hr} BPM!", fg="red", bg="#ffe6e6")
                if last_status != current_status:
                    msg = f"[CRITICAL] Heart Rate {hr} BPM! 🚑 Emergency response initiated!"
                    log_message(msg)
                    speak("Emergency! Heart rate critical. Calling ambulance now.")
                    threading.Thread(target=lambda: winsound.Beep(1000, 800), daemon=True).start()
            elif 120 <= hr <= 160:
                current_status = "high"
                status_label.config(text=f"⚠️ High Heart Rate: {hr} BPM", fg="orange", bg="#fff3cd")
                if last_status != current_status:
                    msg = f"[WARNING] High Heart Rate Detected: {hr} BPM ⚠️"
                    log_message(msg)
                    speak(f"Warning! Heart rate {hr}. Please relax or sit down.")
                    threading.Thread(target=lambda: winsound.Beep(700, 400), daemon=True).start()
            else:
                current_status = "normal"
                status_label.config(text=f"✅ Normal Heart Rate: {hr} BPM", fg="green", bg="#eaffea")
                if last_status != current_status:
                    msg = f"[INFO] Heart rate stable at {hr} BPM ✅"
                    log_message(msg)
                    speak(f"Heart rate normal at {hr} beats per minute.")

            last_status = current_status

    canvas.draw_idle()
    index += 1
    root.after(4000, update_graphs)

# ------------------------------
# Start Dashboard
# ------------------------------
status_label.config(text="System Ready ✅ Monitoring Live Data...", fg="green")
speak("System is ready. Monitoring live heart rate now.")
log_message("[SYSTEM] Dashboard initialized successfully ✅")

root.after(2000, update_graphs)
root.mainloop()
