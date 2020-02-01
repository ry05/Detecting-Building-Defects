"""

Task :
> Images are to be classified as cracks or no cracks (based on pure image processing)
> Output an excel file to understand white/black pixel ratios

Metrics :
> Classification Accuracy
> False Positives
> False Negatives

Areas of improvement :
> How to find that ideal white/black pixel ratio threshold?
> Doesnot work for images with "noise" (wires, locks, doors etc.)
> Every run requires the filename of the excel output to be altered. Try to find a way to work around that.

@author: Ramshankar Yadhunath
"""

# Importing libraries ----------------------------------------------

from tkinter import *
import tkinter as tk
from tkinter import font  as tkfont
from tkinter import filedialog
from PIL import Image, ImageTk

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import xlsxwriter 

# Code for the GUI -------------------------------------------------

class CrackApp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        '''Font of Title in Start Page'''
        self.title_font = tkfont.Font(family='Helvetica', size=48, weight="bold")
        
        # Container into which the frames go

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames={}
        for F in (HomePage, CrackDetector):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Start from the Home Page
        self.show_frame("HomePage")
        
    def show_frame(self,page_name):
        '''Show the Page for the given frame'''
        frame = self.frames[page_name]
        frame.tkraise()
        
''' The Structure of the Home Page '''
class HomePage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent, bg='#ffffff') # setting background to white
        self.controller = controller

        title = Label(self, text="\nTHE WALL CRACK PROJECT", font=('ms serif',36,"bold"), bg="#ffffff")
        title.pack()
        
        subtitle = Label(self, text="The Wall Crack Project is an attempt to bring back the usage of \npure Digital Image Processing" 
                      " to identify cracks in a wall", font=('courier new',16), bg="#ffffff")
        subtitle.pack()
        
        icon = PhotoImage(file="images\home_logo.png")
        image_label = Label(self, image=icon, bg="#ffffff")
        image_label.image = icon
        image_label.place(x=330, y=220)
        
    
        enter = Button(self, text='Enter', font=('courier new', 26),
                              bg='black',
                              fg='white', padx=5, pady=5, width=15,
                              command=lambda: controller.show_frame("CrackDetector"))

        enter.place(x=290,y=550)
     

''' The Structure of the Crack Detector Page '''
class CrackDetector(tk.Frame):
    
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent, bg='#ffffff') # setting background to white
        
        # Initial values
        self.controller = controller
        self.folder=None
        
        # Loads the folder wih images
        def load_folder():
            self.folder = filedialog.askdirectory()
            print(self.folder+" has been loaded")
            load_info["text"]="Folder has been loaded"  # changing the status
         
        # Function to find the largest contour area
        def findGreatestContour(contours):
            areas = []
            largest_area = 0
            largest_contour_index = -1
            i = 0
            total_contours = len(contours)
            while (i < total_contours ):
                area = cv2.contourArea(contours[i])
                areas.append(area)
                if(area > largest_area):
                    largest_area = area
                    largest_contour_index = i
                i+=1
            return largest_area, largest_contour_index, areas
        
        # Function to find and return pre-processed images from an input image
        def image_derivatives(img):
            # Convert to gray
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Morphological Processing - Erosion
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            dilation = cv2.dilate(imgray,se,iterations = 2)
            erosion = cv2.erode(dilation,se,iterations = 2)
            final_img = erosion-imgray

            # Adaptive Thresholding
            ''' Global threshold is used to binarize the image; works on individual images and is much more efficient than local binarization

            blur = cv2.GaussianBlur(final_img,(5,5),0)
            theta,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            binarized = otsu # In OpenCV object detection, object=white & background=black 
            '''
            a_img = img.copy()
            bilat = cv2.bilateralFilter(a_img,25,75,75)
            adapt = cv2.adaptiveThreshold(cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,3)
            binarized = (255-adapt)
            
            # Finding Contours
            ''' A contour is a curve joining all the continuous points(along the boundary), having same color/intensity '''
            image, contours, hierarchy = cv2.findContours(binarized,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # Making a copy as drawContours manipulates the original
            img_copy = img.copy()

            '''
            # Drawing Contours
            crack_highlight = cv2.drawContours(img, contours, -1, (255,0,0), 1) # The last param is the width of the contour (adjust it if you need)
            # Finding largest crack
            largest_area, largest_contour_index, areas = findGreatestContour(contours)
            cnt = contours[largest_contour_index]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            bounding_box = cv2.drawContours(img_copy,[box],0,(0,0,255),2)  
            '''

            # returns the final image with the thresholded image and the cracks highlighted image
            return (binarized)
        
        # Function to return white/black pixel ratio
        def white_black_ratio(img):
            pixel_vals={"white":0, "black":0}
            for i in range(227):
                for j in range(227):
                    if(img[i][j]==0):
                        pixel_vals["black"]+=1
                    else:
                        pixel_vals["white"]+=1
            # return white/black pixel ratio      
            return (pixel_vals["white"]/pixel_vals["black"])
        
        # Classify as crack or no-crack
        def crack_or_no_crack(img):
            if(white_black_ratio(img)<0.01):
                return "No Crack"
            else:
                return "Crack"
            
        # Crack Detector Function
        def crack_detector():
            dirs = os.listdir(self.folder)
            
            # Creating the dictionary for the output excel file
            image_names = []
            white_by_black_ratios = []
            image_class = []
           
            # Making the DIP based classification
            try:
                for item in dirs:
                    full_path = self.folder+'/'+item
                    # append image name
                    image_names.append(item)
                    print(full_path)  
                    if(os.path.isfile(full_path)):
                        img = cv2.imread(full_path)
                        thresholded= image_derivatives(img)
                        # append black/white pixel ratio
                        white_by_black_ratios.append(white_black_ratio(thresholded))
                        # append classification of image
                        image_class.append(crack_or_no_crack(thresholded))
                        
                        # store the cracks' highlighted versions
            except:
                print("Folder has not been loaded!")
              
            # Creating the excel file output
            data = {"Image Name":image_names, "White/Black Pixel Ratio":white_by_black_ratios, "Predicted Class":image_class}
            df = pd.DataFrame(data)
            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = pd.ExcelWriter(self.folder+'/'+'Test_output_adaptive.xlsx', engine='xlsxwriter')
            # Convert the dataframe to an XlsxWriter Excel object.
            df.to_excel(writer, sheet_name='Output_sheet')
            # Close the Pandas Excel writer and output the Excel file.
            writer.save()
            
            class_info["text"]="Classification Complete"  # changing the status

# ----------- Outlining a few things -------------

        title = Label(self, text="\nCRACK DETECTOR", font=('courier new',24,"bold"), bg="#ffffff")
        title.pack()
        
        # Load the folder of images
        load_txt = Label(self, text="Load Image", font=('courier new',16,"bold"), bg="#ffffff")
        load_txt.place(x=60, y=100)
        load_select = Button(self, text="Choose Folder", font=('courier new',10,"bold"), width=50, command=load_folder)
        load_select.place(x=60, y=150)
        
        # Folder Status
        load_info = Label(self, text="No Folder", font=('courier new', 10), bg="#ffffff")
        load_info.place(x=100, y=200)

        # Detect the Crack
        detect_txt = Label(self, text="Detect Crack", font=('courier new',16,"bold"), bg="#ffffff")
        detect_txt.place(x=60, y=250)
        detect_select = Button(self, text="Start Classifying", font=('courier new',10,"bold"), width=50, command=crack_detector)
        detect_select.place(x=60, y=300)
        
        # Classification Status
        class_info = Label(self, text="To be Classified...", font=('courier new', 10), bg="#ffffff")
        class_info.place(x=100, y=350)

        # Output File
        output_file = Label(self, text="Output", font=('courier new',16,"bold"), bg="#ffffff")
        output_file.place(x=60, y=400)
        '''
        output_download = Button(self, text="Download Excel File", font=('courier new',10,"bold"), width=50)
        output_download.place(x=60, y=450)
        '''
        
        # Output Info
        output_info = Label(self, text="Output Location: Inside the Loaded Folder", font=('courier new', 10), bg="#ffffff")
        output_info.place(x=100, y=500)


# Run the GUI ------------------------------------------------------
if __name__ == "__main__":
    app = CrackApp()
    # to set the dimensions of the window
    app.geometry("900x650")
    app.resizable(width=False, height=False)
    app.configure(background='white')
    app.mainloop()
