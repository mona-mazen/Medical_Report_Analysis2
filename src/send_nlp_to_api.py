import requests
import json
import pickle
import os
from src.decision_layer import predict_from_file   # ✅ يشغّل الـ decision layer

API_URL = "https://web-production-fccda.up.railway.app/api/Verification/"
NLP_OUTPUTS_DIR = "nlp_outputs"


def load_nlp_pkl(pid):
    """يحمّل ملف الـ PKL المحفوظ من decision_layer."""
    pkl_path = os.path.join(NLP_OUTPUTS_DIR, f"{pid}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path} — شغّل predict_from_file الأول!")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def send_nlp_to_api(patient_id, report_file_path):
    """
    الخطوات:
    1. يشغّل decision_layer ويحفظ النتايج تلقائياً (CSV + PKL)
    2. يحمّل الـ PKL
    3. يبعت للـ API
    """

    # ── 1. تشغيل الموديل + decision layer + حفظ ──
    pid, final_predictions, alerts = predict_from_file(report_file_path)

    # ── 2. تحميل الـ PKL المحفوظ ──
    nlp_results = load_nlp_pkl(pid)

    # ── 3. تجهيز الـ payload ──
    ai_result = {
        "nlp_output": nlp_results[pid]   # dict: {organ: {prediction, alert}}
    }

    # ── تحديد لو في أي alert خطير ──
    mismatch_alert = any(
        "🚨" in alert_text
        for alert_text in alerts.values()
    )

    payload = {
        "patient":       patient_id,
        "before_scan":   "",
        "after_scan":    "",
        "ai_result":     json.dumps(ai_result, ensure_ascii=False),
        "mismatch_alert": mismatch_alert    # ✅ بيبعت True لو في مشكلة
    }

    # ── 4. بعت الطلب ──
    with open(report_file_path, "rb") as report_file:
        files   = {"nlp_report": report_file}
        response = requests.post(API_URL, data=payload, files=files)

    # ── 5. طباعة النتيجة ──
    print("\n📤 Sent to API:")
    print("  NLP Predictions:", json.dumps(nlp_results[pid], ensure_ascii=False, indent=2))
    print("  Mismatch Alert:", mismatch_alert)
    print("  Status Code:", response.status_code)
    print("  Response:", response.text)

    return response


if __name__ == "__main__":
    patient_id       = 1
    report_file_path = "Data/reports/sample_report.txt"
    send_nlp_to_api(patient_id, report_file_path)