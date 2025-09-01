import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import base64

# ========= 1. ตั้งค่าพื้นฐาน และเริ่มต้น Flask =========
app = Flask(__name__)

IMG_WIDTH, IMG_HEIGHT = 150, 150
MODEL_FILE_NAME = 'weld_model.weights.h5'  # ชื่อไฟล์โมเดลที่ฝึกเสร็จแล้ว

# ========= 2. โหลดโมเดล AI (ทำครั้งเดียวตอนเปิดเซิร์ฟเวอร์) =========
# สร้างโครงสร้างโมเดลเปล่าๆ ให้เหมือนกับตอนที่ฝึกเป๊ะๆ
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# โหลด "น้ำหนัก" (weights) ที่ฝึกไว้แล้วเข้ามาในโมเดล
model.load_weights(MODEL_FILE_NAME)
print(f"--- โหลดโมเดล '{MODEL_FILE_NAME}' เรียบร้อยแล้ว ---")


# ========= 3. สร้างหน้าเว็บ (Routes) =========

# --- หน้าหลัก (Home Page) ---
@app.route('/')
def index():
    # ส่งไฟล์ index.html ในโฟลเดอร์ templates กลับไปให้เบราว์เซอร์
    return render_template('index.html')


# --- ช่องทางสำหรับทำนายผล (Prediction API) ---
@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลรูปภาพที่ถูกส่งมาจากหน้าเว็บ
    data = request.get_json()
    # รูปภาพจะถูกส่งมาในรูปแบบ Base64, เราต้องถอดรหัสกลับมาเป็นรูปภาพ
    img_data = base64.b64decode(data['image'].split(',')[1])

    # แปลงข้อมูลรูปภาพให้ OpenCV และ TensorFlow ใช้งานได้
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # --- ขั้นตอนการประมวลผลรูปภาพ (เหมือนในโค้ดเดิม) ---
    img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    # --- ทำนายผลด้วย AI ---
    prediction_score = model.predict(img_array)[0][0]

    # --- แปลผลลัพธ์ ---
    if prediction_score > 0.5:
        result = "OK"
        confidence = prediction_score * 100
    else:
        result = "NG"
        confidence = (1 - prediction_score) * 100

    # --- ส่งคำตอบกลับไปให้หน้าเว็บในรูปแบบ JSON ---
    return jsonify({'result': result, 'confidence': f'{confidence:.2f}%'})
