from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_image(image_path, method="Gaussian Blur", param1=5, param2=5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if method == "Gaussian Blur":
        processed_img = cv2.GaussianBlur(img, (param1, param1), 0)
    elif method == "Median Blur":
        processed_img = cv2.medianBlur(img, param1)
    elif method == "Canny":
        processed_img = cv2.Canny(img, param1, param2)
    else:
        processed_img = img

    processed_path = os.path.join(PROCESSED_FOLDER, "processed.jpg")
    cv2.imwrite(processed_path, processed_img)
    return processed_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        method = request.form["method"]
        param1 = int(request.form["param1"])
        param2 = int(request.form["param2"])

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            processed_path = process_image(file_path, method, param1, param2)

            return send_file(processed_path, mimetype="image/jpeg")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
