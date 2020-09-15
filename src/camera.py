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

cap = cv2.VideoCapture('src/quickvid.mov')
frameRate = cap.get(5) #frame rate

# cap.set(cv2.CAP_PROP_FPS, 5)
# fps = int(cap.get(5))
# print("fps:", fps)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    gray = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    roi = cv2.resize(gray, (299, 299))/255
    # roi = cv2.rotate(roi, cv2.cv2.ROTATE_90_CLOCKWISE)
    # print(roi[1])
    # new_img = np.delete(roi, 3 , axis = 2).reshape(-1, 299, 299, 3)

    pred = classes[np.argmax(model.predict(roi.reshape(-1, 299, 299, 3)), axis = 1)][0]
    # print(pred)
    # cv2.putText(frame, pred, (50,50) ,font, 1, (40, 40, 0), 2)


    cv2.putText(roi, pred, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 


    cv2.imshow('frame', roi)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
print('cap released')
# cv2.destroyAllWindows('frame')
# print('window destroyed')

