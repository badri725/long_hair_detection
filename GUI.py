

# ## Importing Libraries


import numpy as np 
import pandas as pd 
import cv2 as ocv
import cvlib as cv 
import matplotlib.pyplot as plt
import PIL as im
import tensorflow.keras as tfk
from tensorflow.keras.models import  load_model
import math
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk


# ## Functions for model predictions
#     1. creating function for face detection to find face co-ordinations
#     2. loading model to classify gender and decision making based on classifications
#     3. Two models one for age, gender classification another model for predicting long or short hair 



def picture(im):    
    image = ocv.imread(im) # reading image
    image = ocv.resize(image,(640,640))
    points,acc = cv.detect_face(image) # getting face co-ordinates in given image

    return image,points




def model_out(img,corr):
    gen_model = load_model('Age_gender.keras') #loading age,gender model
    bin_model = load_model('hairDetect.keras') #loading hair detecting model

    gen,age,hair = None,None,None
    if len(corr) == 0:
         messagebox.showinfo("Face Detection Cancelled", "No face detected.")
         return gen,age,hair

    x,y,x_end,y_end = corr[0]
    X,Y= round((x-0)*0.90),round((y-0)*0.90) # calculations for finding boundary of images

    images = img[y-5:y_end+5,x-5:x_end+5] # image cropping
    images = ocv.resize(images,(110,110))
    image = images.astype(np.float32)/255.0
    val = gen_model.predict(np.expand_dims(image,axis=0)) # prediction for age,gender
    gen,age = int(val[0][0][0]>0.5),round(val[1][0][0])

    if age >=20 and age <=30:
        binary = img[y-Y:y_end+70,x-X:x_end+75] # image cropping
        binary = ocv.resize(binary,(110,110))
        binary = ocv.cvtColor(binary,ocv.COLOR_BGR2GRAY) #converting to gray image
        binary =binary.astype(np.float32)/255.0
        hval = bin_model.predict(np.expand_dims(binary,axis=0))
        hair = int(hval[0][0]>0.5)
  
    # Choosing gender based on given task
    
    color = 'green'     
    if hair ==None:
        gen = 'Male' if gen == 0 else 'Female'
    else:
        color = 'red'
        gen = 'Male' if hair == 0 else 'Female'
    return gen,age,color


# ## GUI creation 
#     1. building gui for predicting images
#     2. Dropdown menu, buttons, labels to provide better gui
#     3. custom images for testing model performance


# Create main window
win = Tk()
win.title('Long_Hair_Detection')
win.config(bg='darkgreen')
win.geometry('600x600')

options = ["image.jpg", "image1.jpg", "image2.jpg", "image3.jpg",'image4.jpg','image5.jpg','image6.jpg','image7.jpg','image8.jpg','image9.jpg','image10.jpg']
label = Label(win,bg='green',fg='black',bd=10,relief=RAISED)
label.pack(side='left',padx=10,pady=10)
label2 = Label(win,bg='white',width=400,height=400,bd=5,relief=RAISED)
label2.pack(side='top',padx=10, pady=100)

def open_file_dialog():
    # Open the file dialog
    global file_path
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]  # File type filters
    )
    # Show a message with the selected file path 
    if file_path:
        images(file_path)

        clicked.set('Custom Images')
    else:
        messagebox.showinfo("File Selection Cancelled", "No file selected.")

def get_selected_option():
    selected_option = clicked.get() #default option
    if selected_option:
        
        images(selected_option)
    else:
        messagebox.showinfo("File Selection Cancelled", "No file selected.")
    

def images(filePath):
    image = im.Image.open(filePath) # configure image and text display inside label
    image = image.resize((300,300))
    global imagetk
    imagetk =ImageTk.PhotoImage(image)
    image,points = picture(filePath)
    gen,age,color = model_out(image,points)
    l = f'gender:{gen} Age:{age}'
    label2.config(text=l,
    font=('arial',25,'bold'),
    fg=color,image=imagetk,
    compound='bottom',pady=10)


clicked = StringVar()
# Set default option
clicked.set('Custom Images')

# Create OptionMenu (Dropdown)
dropdown = OptionMenu(label, clicked, *options)
dropdown.config(fg='white',bg='black')

# Add a button to display the selected option
button = Button(label, text="Show Selected", fg='white',bg='black',command=get_selected_option)

button1 = Button(label,
                text='choose file',
                command=open_file_dialog,fg='white',bg='black')

button1.grid(row=0, column=0, padx=10, pady=10)  # First button in row 0, column 0
dropdown.grid(row=1, column=0, padx=10, pady=10)  # Second button in row 0, column 1
button.grid(row=2, column=0, padx=10, pady=10)  # third button in row 0, column 2

off = Button(win,text='Exit',font=('timesnewroman',20),command=win.destroy)
off.pack(side='bottom')

win.mainloop()








