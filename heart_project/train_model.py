import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyttsx3

# ------------------------------
# Load dataset safely
# ------------------------------
try:
    data = pd.read_csv("heart_disease_uci.csv")  # must be in same folder
    data = data.fillna(0)  # replace NaN with 0
    print("✅ Dataset loaded successfully!")
except Exception as e:
    messagebox.showerror("Error", f"Error loading dataset: {e}")
    raise SystemExit

# Handle missing columns safely
for col in ['trestbps', 'chol', 'thalch', 'fbs']:
    if col not in data.columns:
        data[col] = 0  # add column if missing

# Convert safely to lists
heart_rates = data['trestbps'].astype(float).tolist()
chol = data['chol'].astype(float).tolist()
thalach = data['thalch'].astype(float).tolist()
# Convert 'fbs' safely to 0/1 integers
fbs = [int(x) if str(x).replace('.', '', 1).isdigit() else 0 for x in data['fbs']]

# ------------------------------
# Load AI Model
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
# Tkinter UI Setup
# ------------------------------
root = tk.Tk()
root.title("💓 AI-Powered Heart Monitoring Dashboard")
root.geometry("1000x700")
root.config(bg="#eaffea")

tk.Label(root, text="🧠 AI-Powered Heart Rate Monitoring Dashboard",
         font=("Arial", 16, "bold"), bg="#eaffea", fg="#333").pack(pady=10)

status_label = tk.Label(root, text="Status: Initializing...",
                        font=("Arial", 13, "bold"), bg="#eaffea")
status_label.pack(pady=5)

hr_var = tk.StringVar(value="Heart Rate: -- BPM")
tk.Label(root, textvariable=hr_var,
         font=("Arial", 14, "bold"), bg="#eaffea").pack(pady=5)

# ------------------------------
# Voice setup
# ------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 165)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ------------------------------
# Graph setup (4 graphs)
# ------------------------------
fig, axs = plt.subplots(2, 2, figsize=(9, 5))
titles = ["Resting BP (trestbps)", "Cholesterol (chol)",
          "Max Heart Rate (thalach)", "Fasting Blood Sugar (fbs)"]
lines = []

for ax, title in zip(axs.flatten(), titles):
    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Value")
    ax.grid(True)
    line, = ax.plot([], [], linewidth=2)
    lines.append(line)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=15)

# ------------------------------
# Live Update Logic
# ------------------------------
index = 0
bp_data, chol_data, thalach_data, fbs_data = [], [], [], []

def update_data():
    global index
    if index >= len(heart_rates):
        index = 0

    hr = int(heart_rates[index])
    hr_var.set(f"Current Heart Rate: {hr} BPM")

    bp_data.append(hr)
    chol_data.append(chol[index])
    thalach_data.append(thalach[index])
    fbs_data.append(fbs[index])

    if len(bp_data) > 50:
        bp_data.pop(0)
        chol_data.pop(0)
        thalach_data.pop(0)
        fbs_data.pop(0)

    # Update all 4 graphs
    data_groups = [bp_data, chol_data, thalach_data, fbs_data]
    for ax, line, data_list in zip(axs.flatten(), lines, data_groups):
        line.set_data(range(len(data_list)), data_list)
        ax.set_xlim(0, 50)
        ax.set_ylim(min(data_list) - 10, max(data_list) + 10)
    canvas.draw_idle()

    # Predict health condition
    pred = model.predict([[hr]])[0]
    if pred == 2 or hr > 130 or hr < 60:
        status_label.config(text="🚨 CRITICAL ALERT: Seek Immediate Help!", fg="red", bg="#ffe6e6")
        speak(f"Critical alert! Heart rate is {hr} beats per minute! Immediate medical help needed!")
        messagebox.showerror("Emergency Alert", f"Critical Heart Rate Detected: {hr} BPM!\nSeek medical help immediately!")
    elif 90 <= hr <= 120:
        status_label.config(text="⚠️ Warning: Abnormal", fg="orange", bg="#fff3cd")
    else:
        status_label.config(text="✅ Status: Normal", fg="green", bg="#eaffea")

    index += 1
    root.after(2500, update_data)

# ------------------------------
# Start Monitoring
# ------------------------------
status_label.config(text="System Ready ✅", fg="green")
root.after(2000, update_data)
root.mainloop()
