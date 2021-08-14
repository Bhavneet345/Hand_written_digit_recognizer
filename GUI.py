""""Importing the libraries"""
from keras.models import load_model
from tkinter import *
import tkinter as tk
import numpy as np
import win32gui
from PIL import ImageGrab

"""Loading the model"""

model = load_model("Hand_written_digit_CNN.h5")

"""Class App that handles the whole functionality"""


class APP(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Digit-recognizer")
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, height=300, width=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking...", font=("TimesNewRoman", 40))
        self.prediction_button = tk.Button(self, text="Recognize", command=self.recognize)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.canvas.grid(row=0, column=0, sticky=W, pady=2)
        self.label.grid(row=0, column=1, padx=2, pady=2)
        self.clear_button.grid(row=1, column=0, pady=2)
        self.prediction_button.grid(row=1, column=1, padx=2, pady=2)
        self.canvas.bind('<B1-Motion>', self.draw_lines)

    def clear(self):
        self.canvas.delete("all")

    @staticmethod
    def predict(img):
        img = img.resize((28, 28))
        img = img.convert("L")
        img = np.array(img)
        img = img.reshape((1, 28, 28, 1))
        img = img/255.0
        result = model.predict([img])[0]
        return np.argmax(result), max(result)



    def recognize(self):
        HWND = self.canvas.winfo_id()  # to fet the handle of canvas
        rect = win32gui.GetWindowRect(HWND)  # to get the coordinates of the rectangle
        image = ImageGrab.grab(rect)
        digit, accuracy = self.predict(image)
        self.label.configure(text='{},{:.2f}%'.format(digit, accuracy*100))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        return self.canvas.create_oval(self.x-8, self.y-8, self.x+8, self.y+8, fill="black")


if __name__ == '__main__':
    app = APP()
    mainloop()
