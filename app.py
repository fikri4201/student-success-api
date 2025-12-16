from __future__ import annotations

import os

from datetime import datetime, timezone
from math import exp
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from file:// and local networks


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(z: float) -> float:
    # numerically stable-ish for our ranges
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = exp(z)
        return ez / (1.0 + ez)


def predict_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basit ama makul bir tahmin modeli:
    - Notlar ve devam ağırlıklı
    - Ödev sayısı ve çalışma saati destekleyici
    Çıktı: 0-100 skor + başarı olasılığı (0-1)
    """
    midterm = float(payload.get("midterm_grade", 0))
    homework_avg = float(payload.get("homework_avg", 0))
    previous_gpa = float(payload.get("previous_gpa", 0))
    attendance = float(payload.get("attendance", 0))
    homework_count = float(payload.get("homework_count", 0))
    study_hours = float(payload.get("study_hours", 0))

    # clamp inputs
    midterm = clamp(midterm, 0, 100)
    homework_avg = clamp(homework_avg, 0, 100)
    previous_gpa = clamp(previous_gpa, 0, 100)
    attendance = clamp(attendance, 0, 100)
    homework_count = clamp(homework_count, 0, 20)
    study_hours = clamp(study_hours, 0, 30)

    # weighted score (0-100)
    # weights sum ~ 1.0 after normalization
    score = (
        0.30 * midterm +
        0.25 * homework_avg +
        0.25 * previous_gpa +
        0.15 * attendance +
        0.03 * (homework_count / 20.0) * 100.0 +
        0.02 * (study_hours / 30.0) * 100.0
    )
    score = clamp(score, 0, 100)

    # probability of passing via logistic around threshold 65
    # sharper slope around that point
    probability_pass = sigmoid((score - 65.0) / 7.5)

    if score < 60:
        risk_level = "Yüksek Risk"
        msg = "Öğrenci için acil destek/izleme önerilir."
    elif score < 75:
        risk_level = "Orta Risk"
        msg = "Öğrenci takip edilmeli, destek planı faydalı olabilir."
    else:
        risk_level = "Düşük Risk"
        msg = "Öğrenci genel olarak iyi durumda görünüyor."

    return {
        "score": round(score, 2),
        "risk_level": risk_level,
        "probability_pass": round(probability_pass, 4),
        "message": msg,
        "created_at": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    student_name = (data.get("student_name") or "").strip()

    # Minimal validation
    if not student_name:
        return jsonify({"error": "student_name is required"}), 400

    result = predict_score(data)
    result["student_name"] = student_name
    return jsonify(result)


if __name__ == "__main__":
    # Render/Railway/Fly gibi platformlar PORT env değişkeni verir.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
