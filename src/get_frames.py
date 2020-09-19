import cv2
import numpy as np
from tensorflow.keras.models import load_model


def get_video(model, in_file, out_file, rotation = False, **kwargs):
    cap = cv2.VideoCapture(in_file)
    frameRate = cap.get(5) #frame rate
    out = cv2.VideoWriter(out_file, fourcc, 5, (299, 299))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        roi = cv2.resize(frame, (299,299))/255

        pred = classes[np.argmax(model.predict(roi.reshape(-1, 299, 299, 3)), axis = 1)][0]

        if rotation: 
            roi_ = cv2.rotate(roi, cv2.cv2.ROTATE_90_CLOCKWISE)
        else: 
            roi_ = roi

        cv2.putText(roi_, pred, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

        roi_ = (roi_*255).astype('uint8')
        out.write(roi_)
        cv2.imshow('frame', roi_)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    

if __name__ == '__main__':
    #note - when using this file, be very careful about input dimensions and distortion during resize.
    #to check if aligned - you can run stills through predict.py to compare 
    
    model = load_model("CovNet_logs/best_model_5chords.hdf5") #model to load
    classes = np.array(['Am', 'C', 'Dm', 'F', 'G']) #do not change

    #vars for writing on video frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 100)
    fontScale = 4
    color = (1, 1, 0) 
    thickness = 7

    in_file = 'images/videos_to_predict/vertical.mov'
    out_file = 'images/videos/video_predictions/self_vid_2.avi'

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    #function call
    get_video(model, in_file, out_file, rotation = True, font = font,
                org = org, fontScale = fontScale, color = color, 
                thickness = thickness)
