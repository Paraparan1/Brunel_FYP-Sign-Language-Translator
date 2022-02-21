from datetime import time

from flask import Flask,render_template,Response,jsonify,request
import tensorflow.keras as keras
import numpy as np
import cv2

app = Flask(__name__)

translation=[]

def generate_frames():
    model = keras.models.load_model("ASLModelApp_2.h5")

    class_names = ["del", "spc", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                   "R", "S", "T", "U", "V", "W","X", "Y", "Z"]

    img_size = (60, 60)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret, frame = cap.read()

        height, width, channels = frame.shape

        scale_val = width / height

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_resize = cv2.resize(gray, img_size, fx=scale_val, fy=1, interpolation=cv2.INTER_NEAREST)

        img_array = np.asarray(img_resize).reshape(-1, 60, 60, 1)
        img_array = img_array / 255

        prediction = model.predict(img_array)

        prediction_val = np.argmax(prediction)
        prediction_letter = class_names[prediction_val]
        prediction_accuracy = prediction[0][[prediction_val]]

        if prediction_accuracy > .99:
            translation.append(prediction_letter)


        cv2.putText(frame, "Letter= " + prediction_letter, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(frame, "Accuracy= " + str(*prediction_accuracy * 100) + "%", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 0, 0), 2)
        cv2.putText(frame, "Word= " + "".join(translation), (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        (flag,encodedImage) = cv2.imencode(".jpg",frame)
        if not flag:
            continue
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ bytearray(encodedImage)+ b'\r\n')

def convert_tranlsation():
    convertedtranslation = ""
    for i in translation:
        if i == "spc":
            convertedtranslation += (" ")
        elif i == "del":
            convertedtranslation = convertedtranslation.rstrip(convertedtranslation[-1])

        else:
            convertedtranslation += str(i)
    return convertedtranslation


@app.route("/")
def index():
    return render_template("index.html",data = convert_tranlsation())

@app.route("/video")
def translator_feed():
    return Response(generate_frames(),mimetype="multipart/x-mixed-replace;boundary=frame")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)

