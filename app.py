# app.py
# Flask API Sunucusu - Android Uygulaması ile İletişim

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

print("\n" + "="*70)
print("🌐 ÖĞRENCİ BAŞARI TAHMİN API SUNUCUSU")
print("="*70)

try:
    model = joblib.load('student_success_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('model_info.json', 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    print("✅ Model başarıyla yüklendi!")
    print(f"   Model: {model_info['model_name']}")
    print(f"   Versiyon: {model_info['version']}")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    print("   Lütfen önce 'python train_model.py' komutunu çalıştırın!")
    model = None
    scaler = None
    model_info = None

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Öğrenci Başarı Tahmin API Çalışıyor',
        'version': '1.0',
        'project': 'TÜBİTAK 2209-A',
        'author': 'Fikri Özgen'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    if model_info is None:
        return jsonify({'status': 'error', 'message': 'Model bilgisi bulunamadı'}), 500
    return jsonify({'status': 'success', 'model_info': model_info})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Model yüklenmemiş. Lütfen train_model.py çalıştırın.'
            }), 500
        
        data = request.get_json()
        
        required_fields = ['vize_notu', 'odev_ortalamasi', 'devam_orani', 
                          'odev_sayisi', 'calisma_saati', 'onceki_donem_ortalamasi']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Eksik alanlar: {", ".join(missing_fields)}'
            }), 400
        
        input_data = pd.DataFrame([data])[model_info['feature_columns']]
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        success_score = int(prediction_proba[1] * 100)
        
        if success_score >= 70:
            risk_level = 'low'
            risk_text = 'Düşük Risk'
            status = 'success'
            message = 'Başarılı Olma Olasılığı Yüksek'
            icon = '✅'
            recommendation = 'Öğrenci iyi bir performans sergiliyor. Başarılı olacak görünüyor.'
        elif success_score >= 50:
            risk_level = 'medium'
            risk_text = 'Orta Risk'
            status = 'warning'
            message = 'Orta Seviye Risk'
            icon = '⚡'
            recommendation = 'Öğrenci sınırda. Ek destek ve motivasyon gerekebilir.'
        else:
            risk_level = 'high'
            risk_text = 'Yüksek Risk'
            status = 'danger'
            message = 'Başarısız Olma Riski Yüksek'
            icon = '⚠️'
            recommendation = 'Öğrenci acil desteğe ihtiyaç duyuyor. Akademik danışmanlık önerilir.'
        
        print(f"📊 Tahmin: {success_score}/100 - {risk_text}")
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'basarili': bool(prediction),
                'basari_skoru': success_score,
                'basarisiz_olasilik': round(prediction_proba[0] * 100, 2),
                'basarili_olasilik': round(prediction_proba[1] * 100, 2),
                'risk_seviyesi': risk_level,
                'risk_text': risk_text,
                'durum': status,
                'mesaj': message,
                'icon': icon,
                'oneri': recommendation
            },
            'input_data': data
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Tahmin yapılırken hata: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Sunucu Başlatılıyor...")
    print("="*70)
    print("\n📡 Erişim Adresleri:")
    print("   - Yerel: http://localhost:5000")
    print("   - Ağ: http://0.0.0.0:5000")
    print("\n📍 Endpoint'ler:")
    print("   GET  /              : Ana sayfa")
    print("   GET  /health        : Sağlık kontrolü")
    print("   GET  /model_info    : Model bilgileri")
    print("   POST /predict       : Tahmin yap")
    print("\n⏸️  Durdurmak için: Ctrl+C")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)