import base64
import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request

try:
    import face_recognition
except Exception:
    face_recognition = None

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

DATASET_DIR = Path(os.getenv("DATASET_DIR", Path(__file__).parent / "dataset")).resolve()
RECOGNITION_TOLERANCE = float(os.getenv("RECOGNITION_TOLERANCE", "0.5"))
ORB_MATCH_THRESHOLD = float(os.getenv("ORB_MATCH_THRESHOLD", "45"))

known_encodings = []
known_labels = []
known_orb_descriptors = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
orb = cv2.ORB_create(1000)


def load_known_faces() -> None:
    known_encodings.clear()
    known_labels.clear()
    known_orb_descriptors.clear()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    loaded_count = 0

    for student_dir in DATASET_DIR.iterdir():
        if not student_dir.is_dir():
            continue

        label = student_dir.name
        for image_path in student_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            if face_recognition is not None:
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                if not encodings:
                    app.logger.warning("Skipping image with no face: %s", image_path)
                    continue

                known_encodings.append(encodings[0])
                known_labels.append(label)
                loaded_count += 1
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    app.logger.warning("Skipping invalid image: %s", image_path)
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) == 0:
                    app.logger.warning("Skipping image with no face: %s", image_path)
                    continue

                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                roi = gray[y : y + h, x : x + w]
                _, descriptors = orb.detectAndCompute(roi, None)

                if descriptors is None or len(descriptors) == 0:
                    app.logger.warning("Skipping image with no ORB descriptors: %s", image_path)
                    continue

                known_orb_descriptors.append(descriptors)
                known_labels.append(label)
                loaded_count += 1

    engine = "face_recognition" if face_recognition is not None else "opencv_orb_fallback"
    app.logger.info("Loaded %d face templates using %s from %s", loaded_count, engine, DATASET_DIR)


def recognize_with_orb(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return {"status": "fail", "message": "No face detected"}, 200
    if len(faces) > 1:
        return {"status": "fail", "message": "Multiple faces detected"}, 200

    x, y, w, h = faces[0]
    roi = gray[y : y + h, x : x + w]
    _, live_descriptors = orb.detectAndCompute(roi, None)

    if live_descriptors is None or len(live_descriptors) == 0:
        return {"status": "fail", "message": "Face encoding failed"}, 200

    if not known_orb_descriptors:
        return {"status": "fail", "message": "No known faces in dataset"}, 400

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_index = -1
    best_score = float("inf")

    for index, descriptors in enumerate(known_orb_descriptors):
        matches = matcher.match(live_descriptors, descriptors)
        if not matches:
            continue

        average_distance = float(np.mean([match.distance for match in matches]))
        if average_distance < best_score:
            best_score = average_distance
            best_index = index

    if best_index == -1:
        return {"status": "fail", "message": "Face Not Recognized", "confidence": 0.0}, 200

    confidence = float(max(0.0, min(1.0, 1.0 - (best_score / 100.0))))

    if best_score <= ORB_MATCH_THRESHOLD:
        matched_label = known_labels[best_index]
        app.logger.info("ORB matched label=%s score=%.4f confidence=%.4f", matched_label, best_score, confidence)
        return {
            "status": "success",
            "name": matched_label,
            "confidence": round(confidence, 4),
        }, 200

    app.logger.info("ORB face not recognized. score=%.4f confidence=%.4f", best_score, confidence)
    return {
        "status": "fail",
        "message": "Face Not Recognized",
        "confidence": round(confidence, 4),
    }, 200


def decode_input_image() -> Tuple[np.ndarray, str]:
    if "image" in request.files:
        image_bytes = request.files["image"].read()
    elif "file" in request.files:
        image_bytes = request.files["file"].read()
    else:
        payload = request.get_json(silent=True) or {}
        base64_data = payload.get("image")
        if not base64_data:
            return None, "Image data is required"
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception:
            return None, "Invalid base64 image"

    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        return None, "Invalid image data"
    return image, ""


load_known_faces()


@app.route("/")
def home():
    status = "ready"
    return jsonify({
        "message": "Face Recognition API is Running 🚀",
        "status": status,
        "face_recognition_available": face_recognition is not None,
        "recognition_engine": "face_recognition" if face_recognition is not None else "opencv_orb_fallback",
        "dataset_dir": str(DATASET_DIR),
        "known_faces_count": len(known_labels),
    })


@app.route("/reload-faces", methods=["POST"])
def reload_faces():
    load_known_faces()
    return jsonify({
        "message": "Known faces reloaded",
        "count": len(known_labels),
    })


@app.route("/recognize", methods=["POST"])
def recognize_face():
    try:
        if face_recognition is not None and not known_encodings:
            return jsonify({"status": "fail", "message": "No known faces in dataset"}), 400
        if face_recognition is None and not known_orb_descriptors:
            return jsonify({"status": "fail", "message": "No known faces in dataset"}), 400

        image, error_message = decode_input_image()
        if image is None:
            return jsonify({"status": "fail", "message": error_message}), 400

        if face_recognition is None:
            payload, status_code = recognize_with_orb(image)
            return jsonify(payload), status_code

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)

        if len(face_locations) == 0:
            return jsonify({"status": "fail", "message": "No face detected"}), 200
        if len(face_locations) > 1:
            return jsonify({"status": "fail", "message": "Multiple faces detected"}), 200

        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not encodings:
            return jsonify({"status": "fail", "message": "Face encoding failed"}), 200

        test_encoding = encodings[0]
        matches = face_recognition.compare_faces(known_encodings, test_encoding, tolerance=RECOGNITION_TOLERANCE)
        distances = face_recognition.face_distance(known_encodings, test_encoding)
        best_index = int(np.argmin(distances))

        confidence = float(max(0.0, min(1.0, 1.0 - distances[best_index])))

        if matches[best_index]:
            matched_label = known_labels[best_index]
            app.logger.info("Matched label=%s confidence=%.4f", matched_label, confidence)
            return jsonify({
                "status": "success",
                "name": matched_label,
                "confidence": round(confidence, 4),
            }), 200

        app.logger.info("Face not recognized. best_distance=%.4f confidence=%.4f", float(distances[best_index]), confidence)
        return jsonify({
            "status": "fail",
            "message": "Face Not Recognized",
            "confidence": round(confidence, 4),
        }), 200
    except Exception as exception:
        app.logger.exception("Error processing /recognize")
        return jsonify({"status": "fail", "message": "Error processing image", "error": str(exception)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
