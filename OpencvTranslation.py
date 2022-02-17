import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import cv2
import numpy as np
from tensorflow import keras
from tkinter import *

translation = []


def SignLanguageTransaltor():


    model = keras.models.load_model("ASLModel.h5")

    class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                   "U", "V", "W", "X", "Y", "Z"]

    img_size = (60, 60)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret,frame= cap.read()

        height,width,channels = frame.shape

        scale_val = width/height

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_resize = cv2.resize(gray,img_size,fx=scale_val,fy=1,interpolation=cv2.INTER_NEAREST)

        img_array = np.asarray(img_resize).reshape(-1,60,60,1)
        img_array = img_array/255

        prediction = model.predict(img_array)

        prediction_val = np.argmax(prediction)
        prediction_letter = class_names[prediction_val]
        prediction_accuracy = prediction[0][[prediction_val]]

        if prediction_accuracy > .99:
            translation.append(prediction_letter)

        cv2.putText(frame,"Letter= "+prediction_letter,(10,50),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)
        cv2.putText(frame,"Accuracy= "+str(*prediction_accuracy*100)+"%",(10,100),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)
        cv2.putText(frame,"Word= "+"".join(translation),(10,450),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)


        cv2.imshow("Camera",frame)
        cv2.imshow("Image Input",img_resize)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

SignLanguageTransaltor()

    # frame1.update()
    # T.insert("1.0",translation)
#
#
# def backspace():
#     del translation[-1]
#     T.delete("1.0", "end")
#     T.insert("1.0", translation)
#
#
# def space():
#     translation.append(" ")
#     T.delete("1.0", "end")
#     T.insert("1.0", translation)
#
#
# wordGUI = Tk()
# wordGUI.title("Sign Language Translator")
# wordGUI.geometry("1200x550")
#
# frame1 = Frame(wordGUI, width=100, highlightbackground="blue", highlightthickness="3")
# frame1.grid(row=0, column=0, padx=20, pady=20, ipadx=20, ipady=20)
#
# frame2 = Frame(wordGUI, width=500, height=450, highlightbackground="green", highlightthickness="3")
# frame2.grid(row=0, column=5, padx=70, pady=20, ipadx=20, ipady=20)
#
# T = Text(frame1, height=5, width=52)
#
# l = Label(frame1, text="Translation:")
# l.config(font=("Courier", 14))
#
# b1 = Button(frame1, text="Space", command=space)
# b2 = Button(frame1, text="Backspace", command=backspace)
#
# l.pack()
# T.pack()
# b1.pack()
# b2.pack()
#
# T.insert("1.0", translation)
#
# wordGUI.after(2000, SignLanguageTransaltor)
# wordGUI.mainloop()
#
#
#
