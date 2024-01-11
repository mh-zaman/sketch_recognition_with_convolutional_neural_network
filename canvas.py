import os
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.contours import sort_contours
from keras.models import load_model
from PIL import Image, ImageGrab, ImageTk
from skimage.transform import resize

from similarity import orb_sim, structural_sim

print("TensorFlow version:", tf.__version__)


class main:
    def __init__(self):
        self.res = ""
        self.pre = [None, None]
        self.bs = 3.0
        self.pick_image_path = None
        self.root = Tk()
        self.root.title("Drawing Book")
        self.root.resizable(False, False)
        # self.root.overrideredirect(True) # turns off title bar
        # self.root size we want to create
        self.root_height = 600
        self.root_width = 1024

        # getting the full screen height and width
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # calculating the geometry padding
        self.x_cordinate = int((screen_width/2) - (self.root_width/2))
        self.y_cordinate = int((screen_height/2) - (self.root_height/2))

        # canvas size
        self.canvas_width = self.root_width - 52
        self.canvas_height = self.root_height - 150

        self.show_image_or_canvas()

        # Create label
        self.label = Label(
            self.root, text="Pick an image or Sketch here...👆👇", )
        self.label.config(font=("Courier", 14), justify=CENTER)
        self.label.grid(row=1, column=0, columnspan=4)

        # Create label blank for spacing
        space = Label(self.root, text="", )
        space.config(font=("Courier", 14))
        space.grid(row=2, column=0, columnspan=4)

        style = Style()

        ''' Button 1: Exit'''
        style.configure('E.TButton', font=(
            'calibri', 15, 'bold', 'underline'), foreground='red')
        exit_btn = Button(self.root, text='Quit',
                          style='E.TButton', command=self.close)
        exit_btn.grid(row=3, column=0)

        ''' Button 2: Clear'''
        style.configure('C.TButton', font=(
            'calibri', 15, 'bold', 'underline'), foreground='blue')
        calculate_btn = Button(self.root, text='Clear',
                               style='C.TButton', command=self.clear)
        calculate_btn.grid(row=3, column=1)

        ''' Button 3: Detect'''
        style.configure('S.TButton', font=(
            'calibri', 15, 'bold', 'underline'), foreground='green')
        exit_btn = Button(self.root, text='Detect',
                          style='S.TButton', command=self.solve)
        exit_btn.grid(row=3, column=2)

        ''' Button 4: Browse Image'''
        style.configure('B.TButton', font=(
            'calibri', 15, 'bold', 'underline'), foreground='green')
        exit_btn = Button(self.root, text='Browse Image',
                          style='B.TButton', command=self.browse)
        exit_btn.grid(row=3, column=3)

        self.root.mainloop()

    # Funtion for showing image or canvas
    def show_image_or_canvas(self):
        if self.pick_image_path is None:
            print('Showing canvas...')
            self.root.geometry("{}x{}+{}+{}".format(self.root_width,
                                                    self.root_height, self.x_cordinate, self.y_cordinate))
            self.c = Canvas(self.root, bd=3, relief="ridge", bg='white',
                            height=self.canvas_height, width=self.canvas_width)
            self.c.grid(row=0, column=0, columnspan=4, padx=20, pady=10)
            self.c.bind("<Button-1>", self.putPoint)
            # self.c.bind("<ButtonRelease-1>", self.getResult)
            self.c.bind("<B1-Motion>", self.paint)
            pass

        else:
            print('Showing image...')
            img = Image.open(self.pick_image_path)
            img = img.resize((self.canvas_width, self.canvas_height))
            img = ImageTk.PhotoImage(img)
            self.canvas_img = Label(self.root)
            self.canvas_img.configure(image=img)
            self.canvas_img.image = img
            self.canvas_img.grid(
                row=0, column=0, columnspan=4, padx=20, pady=10)
            pass

    # Function for browsing image
    def browse(self):
        fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image", filetypes=(
            ("all files", "*.*"), ("jpg files", "*.jpg"), ("jpeg files", "*.jpeg"), ("png files", "*.png")))
        print('Browsed Image: ', fln)
        self.pick_image_path = fln
        self.show_image_or_canvas()
        self.get_image_and_predict(self.pick_image_path)

    # Function for closing window
    def close(self):
        self.root.destroy()

    # Function for clearing the canvas
    def clear(self):
        self.label['text'] = "Pick an image or Draw Sketch here...👆👇"
        if self.pick_image_path is not None:
            self.pick_image_path = None
            self.show_image_or_canvas()
        else:
            self.c.delete("all")

    # Function for putting a point on the canvas
    def putPoint(self, e):
        self.c.create_oval(e.x - self.bs, e.y - self.bs, e.x +
                           self.bs, e.y + self.bs, outline='black', fill='black')
        self.pre = [e.x, e.y]

    # Function for drawing on the canvas
    def paint(self, e):
        self.c.create_line(self.pre[0], self.pre[1], e.x, e.y,
                           width=self.bs * 2, fill='black', capstyle=ROUND, smooth=TRUE)
        self.pre = [e.x, e.y]

    # Function for solving the prediction
    def solve(self):
        print('Solving the prediction...')
        self.label['text'] = 'Solving the prediction...'

        success = self.get_image()
        if success:
            self.get_image_and_predict("output/2_drawn-image.png")

    # Function for getting the image from the canvas
    def get_image(self):
        print('Processing image from the canvas...')
        self.label['text'] = 'Processing image from the canvas...'

        x, y = (self.c.winfo_rootx(), self.c.winfo_rooty())
        width, height = (self.c.winfo_width(), self.c.winfo_height())
        a, b, c, d = (x, y, x+width, y+height)

        img = ImageGrab.grab()
        img.save("output/1_full-screen.png")
        img = img.crop((a + 76, b + 48, c + 313, d + 154))
        img.save("output/2_drawn-image.png")
        print('Image saved!')
        self.label['text'] = 'Image saved!'
        return True

    # Function for solving the prediction

    def get_image_and_predict(self, path):
        try:
            print('Predicting...')
            self.label['text'] = 'Predicting...'
            model = load_model('sketch_recognition_cnn.keras')

            img = cv2.imread(path)

            ##### removing noise #####
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # blur
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)
            # divide
            divide = cv2.divide(gray, blur, scale=255)
            # otsu threshold
            thresh = cv2.threshold(
                divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            # apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # write result to disk
            cv2.imwrite("output/3_gray_noise_remove.jpg", gray)
            cv2.imwrite("output/4_blur_noise_remove.jpg", blur)
            cv2.imwrite("output/5_divide_noise_remove.jpg", divide)
            cv2.imwrite("output/6_thresh_noise_remove.jpg", thresh)
            cv2.imwrite("output/7_morph_noise_remove.jpg", morph)

            img = cv2.imread("output/6_thresh_noise_remove.jpg")
            # img = cv2.imread("output/7_morph_noise_remove.jpg")

            img = cv2.resize(img, (self.canvas_width, self.canvas_height))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(img_gray, 30, 150)
            cv2.imwrite("output/8_all_canny_detect.jpg", edged)
            contours = cv2.findContours(
                edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            print('Number of contours found: ', len(contours))
            contours = sort_contours(contours, method="left-to-right")[0]
            labels = ['apple', 'banana', 'candle', 'donut', 'envelope',
                    'flower', 'ice_cream', 'leaf', 'mug', 'umbrella']

            area_max = -9999

            for i, c in enumerate(contours):
                print('Processing the image...: ', str(i+1))
                (x, y, w, h) = cv2.boundingRect(c)
                print('x: ', x, 'y: ', y, 'w: ', w, 'h: ', h)
                if x > 0 and y > 0 and w > 20:  # cheaking weather any garbage value detecting
                    roi = img_gray[y:y+h, x:x+w]
                    print(roi)
                    thresh = cv2.threshold(
                        roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    (th, tw) = thresh.shape
                    if tw > th:
                        thresh = imutils.resize(thresh, width=32)
                    if th > tw:
                        thresh = imutils.resize(thresh, height=32)
                    (th, tw) = thresh.shape
                    dx = int(max(0, 32 - tw)/2.0)
                    dy = int(max(0, 32 - th) / 2.0)
                    padded = cv2.copyMakeBorder(
                        thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    padded = cv2.resize(padded, (32, 32))
                    padded = np.array(padded)
                    padded = padded/255.
                    padded = np.expand_dims(padded, axis=0)
                    padded = np.expand_dims(padded, axis=-1)

                    pred = model.predict(padded)
                    pred = np.argmax(pred, axis=1)
                    label = labels[pred[0]]
                    print('>>>>The {} no word is : {}'.format(i, label))

                    area = w * h
                    if (area_max < area):
                        area_max = area
                        prediction = label
                        print("cropping")
                        cropped_image = img[y:y + h, x:x + w]
                        cv2.imwrite(
                            "output/9_cropped_image_of_prediction.png", cropped_image)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, label, (x-5, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("output/10_system_prediction.jpg", img)
            plt.imshow(img)
            plt.savefig('output/11_with_axis.png')
            plt.axis('off')
            plt.savefig('output/12_without_axis.png')
            # Check file paths  -------------Similarity Start----------------

            apple = './content/apple_ref.jpg'
            banana = './content/banana_ref.jpg'
            candle = './content/candle_ref.jpg'
            donut = './content/donut_ref.jpg'
            envelope = './content/envelope_ref.png'
            flower = './content/flower_ref.png'
            ice_cream = './content/ice_cream_ref.jpg'
            leaf = './content/leaf_ref.jpg'
            mug = './content/mug_ref.png'
            umbrella = './content/umbrella_ref.jpg'

            # Dictionary mapping predictions to image paths
            prediction_to_path = {
                "apple": apple,
                "banana": banana,
                "candle": candle,
                "donut": donut,
                "envelope": envelope,
                "flower": flower,
                "ice_cream": ice_cream,
                "leaf": leaf,
                "mug": mug,
                "umbrella": umbrella
            }

            # Check if prediction is in the dictionary
            if prediction in prediction_to_path:
                img1_path = prediction_to_path[prediction]
            else:
                # Handle the case when prediction is not in the dictionary
                print(f"Unknown prediction: {prediction}")
                img1_path = None  # Set img1_path to a default value or handle it accordingly

            img2_path = './output/9_cropped_image_of_prediction.png'

            if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
                print("Image files not found.")
            else:
                img1 = cv2.imread(img1_path, 0)
                img2 = cv2.imread(img2_path, 0)

                if img1 is None or img2 is None:
                    print("Error loading image files.")
                else:
                    # Image similarity calculations
                    orb_similarity = orb_sim(img1, img2)
                    orb_similarity = round(orb_similarity * 100, 2)
                    print("Similarity using ORB is:", orb_similarity)

                    # Resize the second image for SSIM if needed
                    img2_resized = resize(
                        img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
                    ssim = structural_sim(img1, img2_resized)
                    ssim = round(ssim * 100, 2)
                    print("Similarity using SSIM is:", ssim)

                    # -------------Similarity Finish----------------

            self.label['text'] = 'The sketch drawn is {}'.format(
                prediction) + '\nSimilarity using ORB is: {:.2f}% and SSIM is: {:.2f}%'.format(orb_similarity, ssim)

        except Exception as e:
            self.label['text'] = 'Sorry! I got hanged 🤐.\nError: {}'.format(e)
            print('Error: {}'.format(e))
        pass


# Running the main class
if __name__ == "__main__":
    main()
