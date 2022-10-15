
import time
import cv2
import numpy as np
import keyboard as k
import mouse as m
import pyautogui as p
from keras.preprocessing import image 
from model import classes
from keras.models import load_model
from win32gui import FindWindow, GetWindowRect
from cgi import print_environ_usage
from socket import timeout


classify = load_model('classification_model.h5')
#classify = load_model('model1.h5')
direc=[]
predicts=[]
control_done = []
control_currT = 0
control_prevT = 0
curr = 0
prev = 0


def nothing(x):
    pass

def rate(frame):
    global curr, prev
    curr = time.time()
    fps = 1 / (curr - prev)
    prev = curr
    cv2.putText(frame, str(int(fps)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

def prediction(frame):
    hand = frame.copy()
    hand = cv2.resize(hand,(89, 100))
    hand = np.expand_dims(hand, axis = 0)
    hand = np.array(hand, dtype = 'float32')
    #hand = hand.reshape((len(hand), 89, 100, 1))
    prediction = np.argmax(classify.predict(hand.reshape(len(hand), 89, 100, 1)))
    gesture = next(key for key, value in classes.items() if value == prediction)
    print(gesture)
    return(gesture)    

def locate(cm):
    x,y,w,h = cv2.boundingRect(cm)
    #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255,0), 2)
    return (x,y)

def valid(gesture):
    global control_prevT, control_currT
    valid = True
    control_currT = time.time()
 
    if(control_done != []):
        if (control_done[-1]==gesture):
            if((control_currT-control_prevT)<0.5):
             valid = False
        
    print(valid)
    control_prevT = control_currT
    return valid


def control(gesture, direc):
    x = [x[0] for x in direc]
    y = [y[1] for y in direc]
    print(x)
    
    if (gesture == "palm") and valid(gesture):
        if(x[2]<x[10]):
            k.press_and_release('right')
        elif(x[2]>x[10]):
            k.press_and_release('left')

    elif (gesture == "fist") and valid(gesture):
        if(x[2]<x[10]):
            k.press('ctrl')
            k.press('+')
            k.release('ctrl')
            k.release('+')
            print("in fist")
        elif(x[2]>x[10]):
            k.press('ctrl')
            k.press('-')
            k.release('ctrl')
            k.release('-')

    elif (gesture == "peace") and valid(gesture):
       k.press_and_release('ctrl+r')
        
    elif (gesture == "index"):
    #cv2.circle(image, (50, 50), radius=2, color=(0, 0, 255), thickness=-1) 
        window_title = p.getActiveWindowTitle()
        print(window_title)
        _,_,w,h   = GetWindowRect(window_title)
        print(w,h)
        #Xscale = (w/200)
        #Yscale = (h/200)
        for (x,y) in direc: #works for x*6.4, y*3.6
            #p.moveTo(x*Xscale, y*Yscale)
            m.move(x,y)
            

    control_done.append(gesture)



cv2.namedWindow("Color Adjustments",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (400, 300)) 
cv2.moveWindow("Color Adjustments", 50,500)
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

#Color Detection Track
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

time.sleep(2.0)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

i = 0 
    #start = time.time()
while i<100000: 
    ret, frame = cam.read()
    if not ret:
        continue
    frame = cv2.flip(frame,2)
    frame = cv2.resize(frame,(400,200))
    cv2.rectangle(frame, (200,0), (400,200), (255, 0, 0), 2)
    crop_image = frame[0:200, 200:400]

    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])


    rate(frame)
    #Approach 1
    #
    imgMask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtr = cv2.bitwise_and(crop_image, crop_image, mask=imgMask) 
    h,s,v = cv2.split(filtr)
    cv2.imshow("gray", v)
    val = cv2.getTrackbarPos("Thresh", "Color Adjustments") 
    ret, thresh = cv2.threshold(v,val,255,cv2.THRESH_BINARY) 
    blur = cv2.GaussianBlur(thresh,(3,3),0)
    dil = cv2.dilate(blur,(3,3),iterations = 6) 
    contours, hier = cv2.findContours(dil,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Approach 2 
    #
    #imgMask = cv2.inRange(hsv, lower_bound, upper_bound)
    #thresh  = cv2.bitwise_not(imgMask) #to make foreground white and background black
    #blur = cv2.GaussianBlur(thresh,(3,3),0)
    #dil = cv2.dilate(blur,(3,3),iterations = 6) 
    #contours, hier = cv2.findContours(dil,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

   
    try:
        if len(contours) == 0:
            print("No hand")
            direc=[]
            predicts=[]
        else:
            cm = max(contours, key= lambda x: cv2.contourArea(x))  
            cv2.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
            if(i%3==0): 
                gesture = prediction(thresh)
            direc.append(locate(cm))
            predicts.append(gesture)
            if len(predicts)>5 and len(predicts)<30:
            #if len(predicts)==30:
                gesture = max(predicts)
                print(predicts)
                predicts=[]
                control(gesture, direc)
            
            cv2.putText(frame, str(gesture), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
    except:
        pass
     
    
    cv2.imshow("Thresh", thresh)
    cv2.moveWindow("Thresh", 500,50)
    #cv2.imshow("img mask",imgMask)
    #cv2.imshow("filter==", filtr)
    cv2.imshow("Result", frame)
	

    #exit
    key = cv2.waitKey(25) &0xFF    
    if key == 27: 
        break

    i = i+1 


cam.release()
cv2.destroyAllWindows()

#end = time.time() - start
#print("Time taken: " + str(end))





