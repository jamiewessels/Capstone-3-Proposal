import cv2
import numpy as np
from tensorflow.keras.models import load_model


def get_video(model, in_file, out_file, rotation = False, **kwargs):
    '''
        Make predictions on video frames

                Parameters:
                        model: loaded model with loaded weights
                        in_file (str): input video to analyze
                        out_file (str): output video to be saved 
                        rotation (int): rotation angle in degrees; use if video needs to be rotated 
                Returns:
                        displays and saves video with predictions on screen
        '''    
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
    
    model = load_model("CovNet_logs/best_model_5chords.hdf5") #model to load
    classes = np.array(['Am', 'C', 'Dm', 'F', 'G']) #do not change

    #vars for writing on video frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 100)
    fontScale = 4
    color = (1, 1, 0) 
    thickness = 7

    in_file = 'images/videos/videos_to_predict/youtube_cropped.mp4'
    out_file = 'images/videos/video_predictions/choose_output_name.avi' #choose output name

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    #function call
    get_video(model, in_file, out_file, rotation = False, font = font,
                org = org, fontScale = fontScale, color = color, 
                thickness = thickness)
