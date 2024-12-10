from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import sqlite3
import os
from datetime import datetime
import time
from detector import detect_faces, detect_face_attributes
from database_init import add_worker_to_database
import base64
import numpy as np
from PIL import Image
import io
import json

app = Flask(__name__)

# Initialize face detection
cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
model_path = "recognizer/trainingdata.yml"

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Cascade classifier file not found at {cascade_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

facedetect = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Global variables
camera = None
CONFIDENCE_THRESHOLD = 65

def get_camera():
    """Initialize and return camera object"""
    for port in [0, 1, -1]:
        cap = cv2.VideoCapture(port, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap
            cap.release()
    return None

def get_profile(id):
    """Get user profile from database"""
    try:
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.execute("SELECT * FROM Peoples WHERE id=?", (id,))
        profile = cursor.fetchone()
        return profile
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

def log_detection(person_id, person_name):
    """Log face detection to database"""
    try:
        conn = sqlite3.connect("FaceBase.db")
        conn.execute('''CREATE TABLE IF NOT EXISTS DetectionHistory
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        conn.execute("INSERT INTO DetectionHistory (person_id, person_name) VALUES (?, ?)",
                    (person_id, person_name))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def generate_frames():
    """Generate video frames with face detection"""
    global camera
    if camera is None:
        camera = get_camera()
        if camera is None:
            return

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                id, conf = recognizer.predict(face_roi)
                
                if conf < CONFIDENCE_THRESHOLD:
                    profile = get_profile(id)
                    if profile:
                        name = profile[1]
                        info_color = (0, 255, 127)
                        cv2.putText(frame, f"Name: {name}", (x, y+h+30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
                        # Log detection
                        log_detection(id, name)
                else:
                    cv2.putText(frame, "Unknown", (x, y+h+30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                print(f"Recognition error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_employee_page')
def add_employee_page():
    return render_template('add_employee.html')

@app.route('/add_employee', methods=['POST'])
def add_employee():
    try:
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        captured_images = request.form['capturedImages']
        captured_images = json.loads(captured_images)  # Convert JSON string back to list

        # Save images to dataset folder
        dataset_dir = "dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Get next available ID
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM Peoples")
        max_id = cursor.fetchone()[0]
        new_id = 1 if max_id is None else max_id + 1
        
        for i, image_data in enumerate(captured_images):
            image_data = image_data.split(',')[1]  # Get the base64 part
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_path = os.path.join(dataset_dir, f"{new_id}_{name}_{i}.jpg")
            cv2.imwrite(image_path, image)  # Save each image

        # Add to database
        cursor.execute("INSERT INTO Peoples (id, Name, Age, Gender) VALUES (?, ?, ?, ?)",
                      (new_id, name, age, gender))
        conn.commit()
        conn.close()

        return redirect(url_for('video_feed', message='Employee added successfully!'))
    except Exception as e:
        print(f"Error adding employee: {e}")  # Log the error
        return str(e), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_faces')
def detect_faces():
    # This function will handle the detection logic
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_history')
def get_history():
    conn = sqlite3.connect("FaceBase.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dh.detection_time, dh.person_name, p.Age, p.Gender 
        FROM DetectionHistory dh 
        LEFT JOIN Peoples p ON dh.person_id = p.id 
        ORDER BY dh.detection_time DESC LIMIT 10
    """)
    history = cursor.fetchall()
    conn.close()
    
    return jsonify([{
        'time': entry[0],
        'name': entry[1],
        'age': entry[2],
        'gender': entry[3]
    } for entry in history])

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        # Detect faces and add to database
        workers = detect_faces(file_path)
        for worker in workers:
            add_worker_to_database(worker)
        return 'Faces detected and added to database', 200

if __name__ == '__main__':
    app.run(debug=True)
