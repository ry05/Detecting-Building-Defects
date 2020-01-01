"""

Task :
Take in input images, perform image processing functions, store outputs, use them for CNNs

Input Image Size : 227 x 227 pixels

The Outputs will be organized in the following folders as follows :
    1. /bilateral_blur -> Smoothened Images
    2. /otsu_threshold -> Otsu Thresholding (after smoothening)
    3. /adaptive_threshold -> Local Thresholding (after smoothening)
    4. /sharpened -> Image Sharpening (after the other operations)

You can use the images in these folders to train the CNN.
Post that, let's make a comparison of the performance.

@author: Ramshankar Yadhunath
"""

# Importing libraries ----------------------------------------------

from tkinter import *
import tkinter as tk
from tkinter import font  as tkfont
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np

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

        title = Label(self, text="\nCONVERTING IMAGES", font=('ms serif',36,"bold"), bg="#ffffff")
        title.pack()
        
        subtitle = Label(self, text="Preprocessing image input for the CNNs", font=('courier new',16), bg="#ffffff")
        subtitle.pack()        
    
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
            
        # Gabor filtering try
        def build_filters():
            filters = []
            ksize = 31
            for theta in np.arange(0, np.pi, np.pi / 16):
                kern = cv2.getGaborKernel((ksize, ksize), 3.0, theta, 8.0, 30.0, 0, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
            return filters
 
        def process(img, filters):
            accum = np.zeros_like(img)
            for kern in filters:
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
                np.maximum(accum, fimg, accum)
            return accum
        
        def bilateral_blur(img):
            return (cv2.bilateralFilter(img,31,75,75))
        
        def otsu_thresh(img):
            '''
            Performs Otsu Thresholding ->
            1. Bilateral blurring
            2. Conversion to grayscale
            3. Gaussian blurring
            4. Thresholding
            '''
            bilat = cv2.bilateralFilter(img,25,75,75)
            imgray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
            gauss_blur = cv2.GaussianBlur(imgray,(5,5),0)
            theta,otsu = cv2.threshold(gauss_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return (255-otsu)
        
        def adapt_thresh(img):
            '''
            Performs Adaptive Thresholding ->
            1. Bilateral blurring
            2. Conversion to grayscale
            3. Thresholding
            '''
            bilat = cv2.bilateralFilter(img,25,75,75)
            adapt = cv2.adaptiveThreshold(cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,3)
            return (255-adapt)
        
        def sharpen(img):
            ''' 
            Performs the following operations in a sequence ->
            1. Bilateral blurring
            2. Conversion to grayscale
            3. Laplacian Sharpening
            '''
            ddepth = cv2.CV_16S
            kernel_size = 9
            dst = cv2.Laplacian(cv2.cvtColor(bilateral_blur(img), cv2.COLOR_BGR2GRAY), ddepth, ksize=kernel_size)
            return dst
            
        
        # Function to find and return pre-processed images from an input image
        def image_derivatives(img):
            
            bb = bilateral_blur(img)
            ot = otsu_thresh(img)
            ad = adapt_thresh(img)
            sh = sharpen(img)

            # returns the final image with the thresholded image and the cracks highlighted image
            return (bb,ot,ad,sh)     
            
        # Crack Detector Function
        def crack_detector():
            dirs = os.listdir(self.folder)
            
            # Creating the new output folders
                
            bbf = self.folder + "/bilateral_blur"
            try:
                # Create target Directory
                os.mkdir(bbf)
                print("Directory " , bbf ,  " Created ") 
            except FileExistsError:
                print("Directory " , bbf ,  " already exists")
            
            otf = self.folder + "/otsu_threshold"
            try:
                # Create target Directory
                os.mkdir(otf)
                print("Directory " , otf ,  " Created ") 
            except FileExistsError:
                print("Directory " , otf ,  " already exists")
                
            atf = self.folder + "/adaptive_threshold"
            try:
                # Create target Directory
                os.mkdir(atf)
                print("Directory " , atf ,  " Created ") 
            except FileExistsError:
                print("Directory " , atf ,  " already exists")
                
            shf = self.folder + "/sharpened"
            try:
                # Create target Directory
                os.mkdir(shf)
                print("Directory " , shf ,  " Created ") 
            except FileExistsError:
                print("Directory " , shf ,  " already exists")
           
            # Making the conversions and storing
            try:
                for item in dirs:
                    full_path = self.folder+'/'+item
                    print(full_path)  
                    if(os.path.isfile(full_path)):
                        img = cv2.imread(full_path)
                        b,o,a,s = image_derivatives(img)
                        bb = Image.fromarray(b)
                        ot = Image.fromarray(o)
                        at = Image.fromarray(a)
                        sh = Image.fromarray(s) 
                    
                        
                    # store the cracks' highlighted versions
                    bb.save(bbf+'/'+item)
                    ot.save(otf+'/'+item)
                    at.save(atf+'/'+item)
                    sh.save(shf+'/'+(item[:-3]+'png')) #.jpg throws error here; so .png is used
                        
            except:
                print("There has been an error. Check the source code please.")
            
            class_info["text"]="Conversion Complete"  # changing the status

# ----------- Outlining a few things -------------

        title = Label(self, text="\nIMAGE PREPROCESSING", font=('courier new',24,"bold"), bg="#ffffff")
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
        detect_select = Button(self, text="Start Converting", font=('courier new',10,"bold"), width=50, command=crack_detector)
        detect_select.place(x=60, y=300)
        
        # Classification Status
        class_info = Label(self, text="Computing...", font=('courier new', 10), bg="#ffffff")
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
