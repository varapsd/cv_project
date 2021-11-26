import cv2 
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle

import tkinter as tk
from PIL import Image,ImageTk
import time

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

confidence_t=0.99
recognition_t=0.5
required_size = (160,160)

LARGE_FONT = ("Verdana", 25)
cap = cv2.VideoCapture(0)
cap1 = cap



def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    name = 'unknown'
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img, name

class application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        F = PageOne
        frame = F(container,self)
        self.frames[F] = frame
        frame.grid(row=0,column=0, sticky="nsew")
        '''
        for F in (PageTwo,PageOne):
            print(F)
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        '''
        self.show_frame(PageOne)
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.userId = None
        self.userName = None
        self.COUNTER = 0
        self.prev = None
        self.name = None
        self.passTimeInit = None
        self.products = ['product1', 'product2', 'product3', 'product4','product5', 'product6']
        self.productsPrice = [100, 150, 200, 100, 120, 500]
        self.cart = [0 for i in range(len(self.products))]
        self.TOTAL = 0
        label = tk.Label(self, text="Welcome to the Contact Store", font=LARGE_FONT, height=2, bg='skyblue')
        label.pack(pady=10, padx=10, fill=tk.X)
        self.controller = controller
        self.page = 0
        self.frame1 = tk.Frame(self)
        self.frame1.pack(side=tk.LEFT, padx=0, anchor=tk.NW)
        self.webcam = tk.Canvas(self.frame1, width=cv2.CAP_PROP_FRAME_WIDTH * 200,
                                height=cv2.CAP_PROP_FRAME_HEIGHT * 120)
        self.webcam.pack(padx=10)
        self.frame2 = tk.Frame(self)
        self.frame2.pack(side=tk.LEFT,padx=30,anchor=tk.N)
        self.data1 = tk.StringVar()
        self.data = ''
        self.data1.set(self.data)
        self.label1 = tk.Label(self.frame2,textvariable=self.data1,font=('Arial',20,'bold'),bd=15,bg='GreenYellow',width=65,pady=2,justify='center',height=12)
        self.label1.pack(padx=20)
        '''
        button1 = tk.Button(self, text="Back to Home",
                            command=lambda:self.changePage())
        button1.pack()
        '''

        self.passTimeInit = False
        self.passTimePrev = None
        self.passTimeValue = 0
        self.pin = ''
        self.startPage()

    def startPage(self):
        self.data = "------------------------- Capturing Image ------------------------"
        self.data += "\n * Please keep face to the middle of the camera "
        self.data += "\n * Maintain necessary distance to capture the face properly "
        self.data1.set(self.data)
        self.update_image_check()

    def update_image_check(self):
        self.load = cap.read()[1]
        self.image, self.name = detect(self.load,face_detector , face_encoder , encoding_dict)
        self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
        self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
        self.image = Image.fromarray(self.image)  # to PIL format
        self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
        self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)

        if(self.name == 'unknown'):
            self.passTimeInit = None
        elif(self.passTimeInit == None):
            self.passTimeInit = time.time()
            self.prev = self.name
        elif(self.prev != self.name):
            self.prev = self.name
            self.passTimeInit = time.time()
        elif(time.time() - self.passTimeInit >= 5 and self.prev != 'unknown' and self.prev == self.name):
            self.userPage0()
            return
        # Repeat every 'interval' ms
        self.webcam.after(10, self.update_image_check)
    
    def countFingers(self,image):
        with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            totalFinger = None
            if results.multi_hand_landmarks:
                totalFinger = 0
                fingers = [[2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
                i = 0
                for handss in results.multi_hand_landmarks:
                    if (handss.landmark[5].x < handss.landmark[9].x):
                        if (handss.landmark[fingers[0][0]].x > handss.landmark[fingers[0][1]].x and
                                handss.landmark[fingers[0][1]].x > handss.landmark[fingers[0][2]].x):
                            totalFinger += 1
                    else:
                        if (handss.landmark[fingers[0][0]].x < handss.landmark[fingers[0][1]].x and
                                handss.landmark[fingers[0][1]].x < handss.landmark[fingers[0][2]].x):
                            totalFinger += 1
                    for finger in range(1, 5):
                        if (handss.landmark[fingers[finger][0]].y > handss.landmark[fingers[finger][1]].y and
                                handss.landmark[fingers[finger][1]].y > handss.landmark[fingers[finger][2]].y and
                                handss.landmark[fingers[finger][2]].y > handss.landmark[fingers[finger][3]].y):
                            totalFinger += 1
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, str(totalFinger), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

                # cv2.putText(image, str(totalFinger), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            return totalFinger,image
    
    def userPage0(self):
        #playsound('./sounds/transtio.mp3')
        self.data = "------------------------- Capturing Image ------------------------"
        self.data += "\n * Welcome  " + self.name
        self.data +="\n  * Please select the below options to add to cart"
        self.data +="\n   Show all fingers to go to cart and payment"
        self.data += "\n\n\t1. Prodcut1 \t2. Poduct2 "
        self.data += "\n\t3. Prdocut3 \t4. Poduct4 "
        self.data += "\n\t5. Prdocut5 \t6. Poduct6 "
        self.data1.set(self.data)
        self.page0Cam()
    
    def page0Cam(self):
        # Get the latest frame and convert image format
        self.load = cap.read()[1]
        totalFingers,img = self.countFingers(self.load)
        #print(totalFingers)
        self.flag = False
        if(totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers,'confirm')
                self.flag = True
                self.passTimeValue = time.time()+10
                #self.passUpdate(totalFingers)
                print(totalFingers)
                self.optionCheck(totalFingers)
                return
        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            #self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)


        # Repeat every 'interval' ms
        if not self.flag:
            self.webcam.after(10, self.page0Cam)
    
    def optionCheck(self, option):
        if(option == 0):
            self.startPage()
        elif(option == 10):
            self.cartPage()
        elif(option>0 and option<7):
            self.addCart(option-1)

    
    def addCart(self,option):
        self.cart[option] += 1
        print(self.cart)
        self.data = '\n\n\nProduct Added !!'
        self.data1.set(self.data)
        self.label1.after(3000,self.userPage0)
        #self.page0Cam()
    
    def cartPage(self):
        self.data = "------------------------- Capturing Image ------------------------"
        self.data += "\n * Welcome  " + self.name
        self.data +="\nshow all fingers to confirm and payment"
        self.data +="\nProduct \tcount \tprice"
        totalCost = 0
        for i in range(len(self.cart)):
            if(self.cart[i] != 0 ):
                totalCost += self.productsPrice[i]*self.cart[i]
                self.data += "\n"+self.products[i]+" \t"+str(self.cart[i])+"\t"+str(self.productsPrice[i]*self.cart[i])
        self.data += "\n\nTotal cost \t" + str(totalCost)
        self.data += "\n 0. Cancel"
        self.data1.set(self.data)
        self.cartCam()
    
    def cartCam(self):
        self.load = cap.read()[1]
        totalFingers,img = self.countFingers(self.load)
        #print(totalFingers)
        self.flag = False
        if(totalFingers == self.passTimePrev and self.passTimePrev != None):
            if not self.passTimeInit:
                self.passTimeInit = True
                self.passTimeValue = time.time()
            if time.time() - self.passTimeValue >= 2:
                print(totalFingers,'confirm')
                self.flag = True
                self.passTimeValue = time.time()+10
                #self.passUpdate(totalFingers)
                print(totalFingers)
                self.cartCheck(totalFingers)
                return
        else:
            self.passTimeInit = False
            self.passTimePrev = totalFingers
        if not self.flag:
            #self.image = cv2.cvtColor(self.load, cv2.COLOR_BGR2RGB)  # to RGB
            self.image = Image.fromarray(img)  # to PIL format
            self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
            self.webcam.create_image(0, 0, anchor=tk.NW, image=self.image)


        # Repeat every 'interval' ms
        if not self.flag:
            self.webcam.after(10, self.cartPage)

    def cartCheck(self,totalFingers):
        if(totalFingers == 10):
            self.data = "--------- PAYMENT ------------"
            self.data += "\n\n payment confirmed"
            self.data1.set(self.data)
            self.name = None
            self.prev = None
            self.passTimeInit = None
            self.cart = [0 for i in range(len(self.products))]
            self.label1.after(3000,self.startPage)
        elif(totalFingers == 0):
            self.startPage()
        else:
            self.data = "\n--------- INVALID INPUT --------"
            self.data1.set(self.data)
            self.label1.after(3000, self.startPage)

face_encoder = InceptionResNetV2()
path_m = "facenet_keras_weights.h5"
face_encoder.load_weights(path_m)
encodings_path = 'encodings/encodings.pkl'
face_detector = mtcnn.MTCNN()
encoding_dict = load_pickle(encodings_path)

app = application()
width = app.winfo_screenwidth()
height = app.winfo_screenheight()
app.geometry('%dx%d' % (width,height))
app.mainloop()
'''
if __name__ == "__main__":
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        print('yes')
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        
        frame= detect(frame , face_detector , face_encoder , encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
'''

