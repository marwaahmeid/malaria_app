from flask import Flask, render_template, request, redirect, session
import numpy as np
import cv2
import joblib
import os
import json
from datetime import datetime

from utils.feature_extractor import extract_features
import config

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== MODEL ==================
model = joblib.load("models/malaria_stacking_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ================== STATS ==================
def load_stats():
    with open("dashboard/stats.json", "r") as f:
        return json.load(f)

def save_stats(stats):
    with open("dashboard/stats.json", "w") as f:
        json.dump(stats, f, indent=4)


# ================== ROOT (HOME) ==================
@app.route("/")
def home():
    # بعد login نصل مباشرة هنا
    return render_template("index.html")


# ================== LOGIN ==================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session["admin"] = True
            return redirect("/")   # ✅ بعد login → Home
        else:
            return render_template("login.html", error="❌ Wrong credentials")

    return render_template("login.html")


# ================== LOGOUT ==================
@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect("/login")


# ================== PREDICT ==================
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template("index.html", prediction="❌ No image selected")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)

    if img is None:
        return render_template("index.html", prediction="❌ Invalid image")

    # Feature extraction
    features = extract_features(img)
    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    pred = model.predict(features)[0]

    result = "🦠 Parasitized (Malaria Detected)" if pred == 1 else "✅ Uninfected"

    # ================== STATS ==================
    stats = load_stats()

    stats["total"] += 1

    if pred == 1:
        stats["parasitized"] += 1
    else:
        stats["uninfected"] += 1

    stats["history"].append({
        "time": str(datetime.now()),
        "result": result,
        "image": path
    })

    stats["history"] = stats["history"][-10:]

    save_stats(stats)

    return render_template(
        "index.html",
        prediction=result,
        image_path=path
    )


# ================== DASHBOARD ==================
@app.route("/dashboard")
def dashboard():

    if not session.get("admin"):
        return redirect("/login")

    stats = load_stats()
    return render_template("dashboard.html", stats=stats)


# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True)