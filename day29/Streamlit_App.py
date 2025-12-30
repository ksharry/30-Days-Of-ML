# Day 29: Streamlit Web App - AI 影像辨識
# ---------------------------------------------------------
# 這是我們第一個 AI Web App！
# 我們使用 Streamlit 框架，它能讓你用純 Python 寫出漂亮的網頁。
# 這裡我們使用 Keras 內建的 MobileNetV2 (預訓練模型) 來做影像辨識。
# ---------------------------------------------------------

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- 1. 設定網頁配置 (Page Config) ---
st.set_page_config(
    page_title="Day 29 - AI Image Classifier",
    page_icon="📸",
    layout="centered"
)

# --- 2. 載入模型 (Load Model) ---
# 使用 @st.cache_resource 裝飾器，讓模型只載入一次，不用每次重新整理都重跑
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

st.title("📸 Day 29: AI 影像辨識 App")
st.markdown("""
歡迎來到你的第一個 AI App！
請上傳一張照片 (例如：貓、狗、車子、水果)，AI 會告訴你它是什麼。
""")

# 顯示載入中...
with st.spinner('正在載入 AI 模型 (MobileNetV2)...'):
    model = load_model()

# --- 3. 側邊欄 (Sidebar) ---
st.sidebar.header("設定")
confidence_threshold = st.sidebar.slider("信心門檻 (Confidence Threshold)", 0.0, 1.0, 0.2, 0.05)

# --- 4. 上傳圖片 (File Uploader) ---
uploaded_file = st.file_uploader("請選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 顯示圖片
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='上傳的圖片', use_column_width=True)
    
    # --- 5. 影像預處理 (Preprocessing) ---
    # MobileNetV2 需要 224x224 的輸入
    img = image_pil.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # 正規化 (-1 ~ 1)

    # --- 6. 進行預測 (Prediction) ---
    if st.button('開始辨識'):
        with st.spinner('AI 正在思考中...'):
            preds = model.predict(x)
            # 解碼預測結果 (取得前 3 名)
            decoded_preds = decode_predictions(preds, top=3)[0]
            
            st.success("辨識完成！")
            
            # --- 7. 顯示結果 (Display Results) ---
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                if score >= confidence_threshold:
                    st.write(f"**#{i+1}: {label}** ({score*100:.2f}%)")
                    st.progress(float(score))
                else:
                    st.write(f"#{i+1}: {label} (信心度低於門檻)")
else:
    st.info("請上傳圖片以開始。")

# --- 8. 頁尾 (Footer) ---
st.markdown("---")
st.markdown("Made with ❤️ by [30-Days-Of-ML](https://github.com/ksharry/30-Days-Of-ML)")
