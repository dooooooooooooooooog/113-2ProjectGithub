import requests
import cv2
import numpy as np
import pickle
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern

# —— 先实例化 Flask 应用 —— #
app = Flask(__name__)
CORS(app)

# 快取最后一次分析结果
last_result = None

# 载入离线训练好的材质模型
model = pickle.load(open("material_model.pkl", "rb"))

# —— HSV 色系分类表 —— #
HSV_CATEGORIES = {
    "01": {"h": [(0, 10), (170, 179)], "s": (100, 255), "v": (50, 255)},  # 紅
    "02": {"h": [(11, 20)],             "s": (100, 255), "v": (50, 255)},  # 橘
    "03": {"h": [(21, 30)],             "s": (100, 255), "v": (50, 255)},  # 黃
    "04": {"h": [(31,  85)],            "s": ( 50, 255), "v": (50, 255)},  # 綠
    "05": {"h": [(86, 100)],            "s": (100, 255), "v": (50, 255)},  # 青
    "06": {"h": [(101,130)],            "s": (100, 255), "v": (50, 255)},  # 藍
    "07": {"h": [(131,160)],            "s": ( 50, 255), "v": (50, 255)},  # 紫
    "08": {"h": [(161,169)],            "s": (100, 255), "v": (50, 255)},  # 粉
    "09": {"h": [(10,30)],              "s": ( 50,100), "v": (  0,100)},  # 棕
    "10": {"h": [(0,179)],              "s": (  0,  50), "v": (  0, 255)},  # 灰
    "11": {"h": [(0,179)],              "s": (  0,  50), "v": (200, 255)}, # 白
    "12": {"h": [(0,179)],              "s": (  0,  50), "v": (  0,  50)},  # 黑
}

def classify_hsv_color(h, s, v):
    """將單一 HSV 值分類到 12 色系之一"""
    for code, rng in HSV_CATEGORIES.items():
        if rng["s"][0] <= s <= rng["s"][1] and rng["v"][0] <= v <= rng["v"][1]:
            for (hmin, hmax) in rng["h"]:
                if hmin <= h <= hmax:
                    return code
    return None

def read_hsv_from_url(url):
    """下載圖片並轉成 HSV ndarray"""
    resp = requests.get(url)
    arr  = np.frombuffer(resp.content, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def get_dominant_colors_hsv(url, n_clusters=8, max_colors=5):
    """用 KMeans 在 HSV 空間聚類，並 map 到色系 code"""
    hsv    = read_hsv_from_url(url)
    pixels = hsv.reshape(-1,3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)

    codes = []
    for (h,s,v) in centers:
        code = classify_hsv_color(h, s, v)
        if code and code not in codes:
            codes.append(code)
            if len(codes) >= max_colors:
                break
    return codes

def predict_material_url(url, P=8, R=1):
    """用 LBP + SVM 模型預測材質 code"""
    hsv = read_hsv_from_url(url)
    # 轉回灰階做紋理
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P, R, method="uniform")
    hist,_ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, P+3),
        range=(0, P+2)
    )
    hist = hist.astype("float") / hist.sum()
    code = model.predict([hist])[0]
    return str(code)

# —— 路由 —— #

@app.route("/analyze", methods=["POST"])
def analyze():
    global last_result
    try:
        data = request.get_json(force=True)
        url  = data.get("imageUrl")
        if not url:
            return jsonify(error="missing imageUrl"), 400

        colors   = get_dominant_colors_hsv(url, n_clusters=12, max_colors=5)
        material = predict_material_url(url)

        last_result = {"colors": colors, "material": material}
        return jsonify(last_result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@app.route("/result", methods=["GET"])
def result():
    if last_result is None:
        return jsonify(error="no result"), 404
    return jsonify(last_result), 200

# —— 程式進入點 —— #
if __name__ == "__main__":
    print("🚀 啟動分析伺服器，監聽 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
