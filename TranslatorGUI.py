from tkinter import *
import cv2

def backspace():
    del translation[-1]
    T.delete("1.0","end")
    T.insert("1.0", translation)

def space():
    translation.append(" ")
    T.delete("1.0", "end")
    T.insert("1.0", translation)

#List of translations
translation = ["H","E","L","L","O"]
wordGUI = Tk()
# specify size of window.

wordGUI.title("Sign Language Translator")
wordGUI.geometry("1200x550")

frame1=Frame(wordGUI,width=100,highlightbackground="blue",highlightthickness="3")
frame1.grid(row=0,column=0,padx=20,pady=20,ipadx=20,ipady=20)

frame2=Frame(wordGUI,width=500,height=450,highlightbackground="green",highlightthickness="3")
frame2.grid(row=0,column=5,padx=70,pady=20,ipadx=20,ipady=20)

T = Text(frame1, height = 5, width = 52)

l = Label(frame1, text ="Translation:")
l.config(font =("Courier", 14))


b1 = Button(frame1, text ="Space",command=space)
b2 = Button(frame1, text ="Backspace",command=backspace)


l.pack()
T.pack()
b1.pack()
b2.pack()

T.insert("1.0",translation)

wordGUI.mainloop()
