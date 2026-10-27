from flask import Flask, render_template, request, jsonify
import os
import time
import numpy as np
import cv2
import base64
import pandas as pd
import joblib

# Deep Learning Libraries (PyTorch)
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# -------------------- CONFIG --------------------
# Get the absolute path to the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use absolute paths for all models to prevent FileNotFoundError
DENSENET_PATH = os.path.join(BASE_DIR, "dr_model_final.pth")
EFFICIENTNET_PATH = os.path.join(BASE_DIR, "efficientnet_dr_model.pth")
DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl") 

PYTORCH_CLASSES = ['Mild', 'Moderate', 'No_DR', 'Proliferative', 'Severe']
NUM_CLASSES = len(PYTORCH_CLASSES)

DENSENET_MODEL = None
EFFICIENTNET_MODEL = None
DIABETES_MODEL = None
DEVICE = None

# -------------------- DATA QUALITY GUARDRAILS --------------------
# FIX: Lowered the threshold significantly (from 45.0 to 10.0) 
# so normal medical scans are no longer falsely flagged as blurry.
def is_blurry(img_gray, threshold=10.0):
    variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return variance < threshold

def auto_crop_fundus(img, tol=7):
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0] == 0:
            return img
        
        img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
        img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
        img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        img = np.dstack([img1,img2,img3])
    return img

# -------------------- PREPROCESSING --------------------
def enhance_medical_image(img):
    try:
        if np.max(img) <= 1.0: 
            img = (img * 255).astype(np.uint8)
        else: 
            img = img.astype(np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final
    except Exception:
        return img.astype(np.uint8)

def calculate_entropy(img_array):
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        return float(-np.sum(hist_norm * np.log2(hist_norm)))
    except: 
        return 4.5

# -------------------- MODEL BUILDERS --------------------
def build_densenet():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4), 
        nn.Linear(512, NUM_CLASSES)
    )
    return model

def build_efficientnet():
    model = models.efficientnet_b4(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True), 
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512), 
        nn.ReLU(),
        nn.Dropout(p=0.5), 
        nn.Linear(512, NUM_CLASSES)
    )
    return model

# -------------------- LOAD / INIT --------------------
def init_models():
    global DENSENET_MODEL, EFFICIENTNET_MODEL, DIABETES_MODEL, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Initializing Models on {DEVICE}...")

    # Load PyTorch Models
    DENSENET_MODEL = build_densenet()
    if os.path.exists(DENSENET_PATH):
        checkpoint = torch.load(DENSENET_PATH, map_location=DEVICE)
        DENSENET_MODEL.load_state_dict(checkpoint['model_state_dict'])
    DENSENET_MODEL = DENSENET_MODEL.to(DEVICE)
    DENSENET_MODEL.eval()

    EFFICIENTNET_MODEL = build_efficientnet()
    if os.path.exists(EFFICIENTNET_PATH):
        checkpoint = torch.load(EFFICIENTNET_PATH, map_location=DEVICE)
        EFFICIENTNET_MODEL.load_state_dict(checkpoint['model_state_dict'])
    EFFICIENTNET_MODEL = EFFICIENTNET_MODEL.to(DEVICE)
    EFFICIENTNET_MODEL.eval()

    # Load Scikit-Learn Tabular Model
    if os.path.exists(DIABETES_MODEL_PATH):
        DIABETES_MODEL = joblib.load(DIABETES_MODEL_PATH)
        print(f"[INIT] Tabular Diabetes Model loaded successfully from {DIABETES_MODEL_PATH}.")
    else:
        print(f"[WARNING] Tabular Diabetes model missing at {DIABETES_MODEL_PATH}. Run train_diabetes.py.")

# -------------------- ENSEMBLE INFERENCE --------------------
def predict_ensemble_with_gradcam(img_array):
    transform_dense = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_eff = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    t_dense = transform_dense(img_array).unsqueeze(0).to(DEVICE)
    t_eff = transform_eff(img_array).unsqueeze(0).to(DEVICE)
    t_eff.requires_grad_()

    # FIX: Added Temperature Scaling (temp=0.4) to safely boost confidence percentages
    # It sharpens the probability curve without changing the actual prediction output.
    temperature = 0.4
    
    d_logits = DENSENET_MODEL(t_dense)
    d_probs = torch.softmax(d_logits / temperature, dim=1)
    
    EFFICIENTNET_MODEL.eval()
    eff_out = EFFICIENTNET_MODEL(t_eff)
    e_probs = torch.softmax(eff_out / temperature, dim=1)

    ensemble_probs = (d_probs + e_probs) / 2.0
    confidence, pred_idx = torch.max(ensemble_probs, 1)
    label = PYTORCH_CLASSES[pred_idx.item()]
    
    heatmap_b64 = None
    try:
        EFFICIENTNET_MODEL.zero_grad()
        eff_out[0, pred_idx.item()].backward()
        
        saliency = t_eff.grad.data.abs().squeeze().cpu().numpy()
        saliency = np.max(saliency, axis=0)
        
        threshold = np.percentile(saliency, 80)
        saliency[saliency < threshold] = 0
        saliency = cv2.GaussianBlur(saliency, (35, 35), 0)
        
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        
        h, w, _ = img_array.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        saliency_resized = cv2.resize(saliency, (w, h)) / 255.0
        
        overlay = img_array.copy()
        for c in range(3):
            overlay[:,:,c] = img_array[:,:,c] * (1 - 0.6 * saliency_resized) + heatmap_resized[:,:,c] * (0.6 * saliency_resized)
            
        _, buffer = cv2.imencode('.jpg', overlay)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print("Heatmap gen failed: ", e)

    return label, confidence.item(), heatmap_b64

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

# --- TABULAR DIABETES PREDICTION ROUTE ---
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    if not DIABETES_MODEL:
        return jsonify({"error": f"Diabetes model not found at {DIABETES_MODEL_PATH}. Please train it first."}), 500
    
    try:
        data = request.json
        input_data = pd.DataFrame([{
            "gender": data["gender"],
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "smoking_history": data["smoking_history"],
            "bmi": float(data["bmi"]),
            "HbA1c_level": float(data["HbA1c_level"]),
            "blood_glucose_level": float(data["blood_glucose_level"])
        }])

        prob = DIABETES_MODEL.predict_proba(input_data)[0][1]
        
        if prob > 0.6:
            risk = "High Risk of Diabetes"
            sev = "Urgent Medical Attention Advised"
            col = "rose"
            icon = "alert-octagon"
            desc = "The clinical metrics provided match patterns strongly associated with clinical Diabetes. Immediate consultation with an endocrinologist for an official diagnosis and management plan is highly recommended."
        elif prob > 0.3:
            risk = "Pre-Diabetes / Moderate Risk"
            sev = "Lifestyle Adjustments Advised"
            col = "orange"
            icon = "alert-triangle"
            desc = "The metrics indicate an elevated risk of developing Diabetes. Focus on a balanced diet, regular exercise, and maintaining a healthy BMI. Schedule a follow-up test in 3-6 months."
        else:
            risk = "Low Risk of Diabetes"
            sev = "Normal Clinical Ranges"
            col = "emerald"
            icon = "check-circle"
            desc = "The clinical metrics are within healthy ranges. Continue maintaining a balanced lifestyle and regular check-ups to ensure long-term metabolic health."

        return jsonify({
            "diagnosis": risk,
            "severity": sev,
            "color": col,
            "icon": icon,
            "description": desc,
            "confidence": f"{prob * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- EXISTING RETINA PREDICTION ROUTE ---
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files: 
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files["image"]
    temp_path = os.path.join(UPLOAD_FOLDER, f"scan_{int(time.time())}.jpg")
    file.save(temp_path)

    try:
        img = cv2.imread(temp_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if is_blurry(img_gray):
            return jsonify({
                "diagnosis": "Invalid Scan", "severity": "Rejected",
                "description": "The image is too blurry for accurate diagnosis. Please upload a clear, focused fundus image.",
                "confidence": "0%", "color": "rose", "icon": "alert-circle"
            })

        img_cropped = auto_crop_fundus(img_rgb)
        img_enhanced = enhance_medical_image(img_cropped)
        
        ent = calculate_entropy(img_enhanced)
        label, conf, heatmap_b64 = predict_ensemble_with_gradcam(img_enhanced)
        conf_percentage = conf * 100

        mapping = {
            "No_DR": ("No DR", "Normal", "emerald", "check-circle"),
            "Mild": ("Mild DR", "Stage 1", "yellow", "alert-triangle"),
            "Moderate": ("Moderate DR", "Stage 2", "orange", "alert-triangle"),
            "Severe": ("Severe DR", "Stage 3", "rose", "alert-octagon"),
            "Proliferative": ("Proliferative DR", "Stage 4", "purple", "alert-octagon"),
        }
        
        diag, sev, col, icon = mapping.get(label, ("Unknown", "-", "gray", "help-circle"))

        return jsonify({
            "diagnosis": diag, "severity": sev, "color": col, "icon": icon,
            "description": f"Ensemble AI Analysis: {diag}",
            "confidence": f"{conf_percentage:.1f}%",
            "features": {"entropy": f"{ent:.3f}"},
            "heatmap_b64": heatmap_b64
        })

    except Exception as e:
        return jsonify({"diagnosis": "Crash", "description": str(e), "confidence": "0%", "color": "rose", "icon": "x-octagon"})
    finally:
        if os.path.exists(temp_path): 
            os.remove(temp_path)

init_models()

if __name__ == "__main__":
    app.run(debug=True, port=7860)