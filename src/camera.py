import cv2
import numpy as np
from tensorflow.keras.models import load_model



model = load_model("CovNet_logs/Checkpoint7.hdf5")
font = cv2.FONT_HERSHEY_SIMPLEX
org = (100, 100)
fontScale = 4
color = (1, 1, 0) 
thickness = 7
classes = np.array(['C', 'F', 'G'])

cap = cv2.VideoCapture('src/quickvid2.MOV')
frameRate = cap.get(5) #frame rate


fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc_ = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi', fourcc_, 5, (299, 299))

# cap.set(cv2.CAP_PROP_FPS, 5)
# fps = int(cap.get(5))
# print("fps:", fps)
count = 0
pred = ''

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    #figure out which roi to use depending if needs rotated
    
    gray = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    roi = cv2.resize(gray, (299, 299))/255
    
    # roi = cv2.resize(frame, (299,299))/255

    # pred = classes[np.argmax(model.predict(roi.reshape(-1, 299, 299, 3)), axis = 1)][0]
    count += 1
    if count == 2 or count % 15 ==0: #create a lag between predictions
        pred = classes[np.argmax(model.predict(roi.reshape(-1, 299, 299, 3)), axis = 1)][0]

    # proba = model.predict(roi.reshape(-1, 299, 299, 3))
    # print(count, proba)
    # pred = classes[np.argmax(proba, axis = 1)][0]

    cv2.putText(roi, pred, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 


    roi = (roi*255).astype('uint8')
    out.write(roi)
    cv2.imshow('frame', roi)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
print('cap released')
# cv2.destroyAllWindows('frame')
# print('window destroyed')

