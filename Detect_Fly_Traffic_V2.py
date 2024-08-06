'''
Detect flies in video, determine if single or multiple flies, and location, area, multiple fly detection state

1. Resize the video to speed up processing
2. Get the median brightness of the video frames by averaging several (e.g., 20) random video frames
3. For each video frame;
* difference each video frame with the median frame
* blur the difference frame
* apply a fixed threshold to the blurred difference frame
* detect objects 
* save bounding box dimensions of objects with sufficient area

OPERATION
Press 'q' to quit
Use the '+' and '-' keys to change object detect threshold by 1
Old shift while pressing '+' or '-' to change threshould by 10

Thomas Zimmerman IBM Research-Almaden 

v0 6.30.24 Original fly maze code
v1 7.31.24 No resize
v2 7.31.24 Crop just traffic lanes

'''
import numpy as np
import cv2


###############PUT YOUR VIDEO LINK HERE ###################
#vid=r'C:\Users\820763897\Documents\AAA_WORK\FlyShock\flyMaze1_640.mp4'
vid=r'C:\Users\820763897\Documents\AAA_WORK\FlyShock\Video\WIN_20240730_13_25_00_Pro.mp4'
detectFileName='test.csv' # file that saves object data

########## PROGRAM VARIABLES ################################################
medianFrames=100 # number of random frames to calculate median frame brightness
#medianFrames=1 # number of random frames to calculate median frame brightness
skipFrames=1  # give video image autobrightness (AGC) time to settle
BLUR=7          # blur differenced images to remove holes in objectes
THRESH=30       # apply threshold to blurred object to create binary detected objects
DELAY=10
THICK=2         # bounding rectangle thickness

X_REZ=640; Y_REZ=480;   # viewing resolution
MIN_AREA=10             # min area of object detected
MAX_AREA=800             # max area of bject detected
DISPLAY_REZ=(640,480)   # display resolution
PROCESS_REZ=(320,240)   # processing resolution, reduce size to speed up processing
CROP_XY=[340,270]
CROP_WH=[350,200]
print('Process Resolution',PROCESS_REZ)

############# DETECT OUTPUT ##################
detectHeader= 'FRAME,ID,XC,YC,AREA,MULTI_FLAG'
MAX_COL=6
FRAME,ID,XC,YC,AREA,MULTI_FLAG=range(MAX_COL)
detectArray=np.empty((0,MAX_COL), dtype='int') # cast as int since most features are int and it simplifies usage

def crop(im):
    x0=CROP_XY[0]; y0=CROP_XY[1]; x1=x0+CROP_WH[0]; y1=x0+CROP_WH[1]; 
    im=im[y0:y1,x0:x1]
    return(im)

def getMedian(vid,medianFrames,PROCESS_REZ):
    # Open Video
    print ('openVideo:',vid)
    cap = cv2.VideoCapture(vid)
    maxFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('averaging frames:',medianFrames,'total frames:',maxFrame)
     
    # Randomly select N frames
    print('calculating median')
    frameIds = skipFrames+ (maxFrame-skipFrames) * np.random.uniform(size=medianFrames)
    frames = [] # Store selected frames in an array
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, colorIM = cap.read()
        colorIM=crop(colorIM)
        #colorIM=cv2.resize(frame,PROCESS_REZ)
        grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)
        frames.append(grayIM)
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)     # Calculate the median along the time axis
     
    cap.release()
    return(medianFrame)

     

######### MAIN PROGRAM #############

print("\n\nUse '+' and '-' keys to change object detect threshold by 1")
print("Hold shift while pressing '+' or '-' to change threshould by 10\n")
# create median frame
medianFrame=getMedian(vid,medianFrames,PROCESS_REZ)

cap = cv2.VideoCapture(vid)
cap.set(cv2.CAP_PROP_POS_FRAMES, skipFrames) # start movie past skipFrames
frameCount=skipFrames
while(cap.isOpened()):
    
    # read key, test for 'q' quit
    key=chr(cv2.waitKey(DELAY) & 0xFF) # pause x msec
    
    if key== 'q':
        break
    elif key=='=':
        THRESH+=1
        print('Thresh:',THRESH)
    elif key=='+':
        THRESH+=10
        print('Thresh:',THRESH)
    elif key=='-' and THRESH>1:
        THRESH-=1
        print('Thresh:',THRESH)
    elif key=='_' and THRESH>11:
        THRESH-=10    
        print('Thresh:',THRESH)
    
    # get image
    ret, colorIM = cap.read()
    if not ret: # check to make sure there was a frame to read
        break
    frameCount+=1
    colorIM=crop(colorIM)
    # capture frame, subtract meadian brightness frame, apply binary threshold
    #colorIM=cv2.resize(colorIM,PROCESS_REZ)
    grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)    # convert color to grayscale image
    diffIM = cv2.absdiff(grayIM, medianFrame)   # Calculate absolute difference of current frame and the median frame           
    blurIM = cv2.blur(diffIM,(BLUR,BLUR))
    ret,binaryIM = cv2.threshold(blurIM,THRESH,255,cv2.THRESH_BINARY) # threshold image to make pixels 0 or 255
    
    # get contours  
    contourList, hierarchy = cv2.findContours(binaryIM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # all countour points, uses more memory
    
    # draw bounding boxes around objects
    objCount=0      # used as object ID in detectArray
    multiFlag=0     # set if multi objects detected
    for objContour in contourList:                  # process all objects in the contourList
        area = int(cv2.contourArea(objContour))     # find obj area        
        PO = cv2.boundingRect(objContour)
        x0=PO[0]; y0=PO[1]; x1=x0+PO[2]; y1=y0+PO[3]
        if area>MIN_AREA and area<MAX_AREA:         # only detect objects with acceptable area       
            cv2.rectangle(colorIM, (x0,y0), (x1,y1), (0,255,0), THICK) # place GREEN rectangle around each object, BGR
                #print('frame:',frameCount,'objCount:',objCount,'Single Area:',area)
            #cv2.rectangle(binaryIM, (x0,y0), (x1,y1), 255, THICK) # place white rectangle around each object
            xc=int((x1-x0)/2 + x0);  yc=int((y1-y0)/2 + y0);
            # save object parameters in detectArray in format FRAME=0; ID=1;  X0=2;   Y0=3;   X1=4;   Y1=5;   XC=6;   YC=7; CLASS=8; AREA=9; AR=10; ANGLE=11; MAX_COL=12
            parm=np.array([[frameCount,objCount,xc,yc,area,multiFlag]],dtype='int') # create parameter vector (1 x MAX_COL) 
            detectArray=np.append(detectArray,parm,axis=0)  # add parameter vector to bottom of detectArray, axis=0 means add row
            objCount+=1                                     # indicate processed an object
        elif area>MAX_AREA:
            cv2.rectangle(colorIM, (x0,y0), (x1,y1), (0,0,255), THICK) # place RED rectangle around each object, BGR
            print('max area exceeded:',area)
        elif area<MIN_AREA:
           cv2.rectangle(colorIM, (x0,y0), (x1,y1), (255,0,0), THICK) # place BLUE rectangle around each object, BGR
           print('min area exceeded:',area)
      
            #=print('frame:',frameCount,'objCount:',objCount,'Multi  Area:',area)

            
    # shows results
    cv2.imshow('colorIM', cv2.resize(colorIM,DISPLAY_REZ))      # display image
    cv2.imshow('blurIM', cv2.resize(blurIM,DISPLAY_REZ))        # display thresh image
    cv2.imshow('diffIM', cv2.resize(diffIM,DISPLAY_REZ))        # display thresh image
    cv2.imshow('medianFrame', cv2.resize(medianFrame,DISPLAY_REZ))        # display thresh image
    cv2.imshow('binaryIM', cv2.resize(binaryIM,DISPLAY_REZ))    # display thresh image

if frameCount>0:
    print('Done with video. Saving feature file and exiting program')
    np.savetxt(detectFileName,detectArray,header=detectHeader,delimiter=',', fmt='%d')
    cap.release()
else:
    print('Count not open video',vid)
cv2.destroyAllWindows()








