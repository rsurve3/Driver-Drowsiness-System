from keras.preprocessing import image as keras_image
from keras.models import model_from_json
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import dlib
from playsound import playsound

# Predict Facial Landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('trained_model/trained_data.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# print(lStart, lEnd)
# print(rStart, rEnd)


def showImage(img, weight, height):
    screen_res = weight, height
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def predictImage(img, model): # If iamge is opened or closed by comparing it with matrix
    img = np.dot(np.array(img, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    return classes[0][0]


def predictFacialLandmark(img, detector): # Eye landmarks
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    return rects


def readImage(img_path): 
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def getRoi(eye, image): #Calculate Eye Image Ratio
    (x, y, w, h) = cv2.boundingRect(np.array([eye]))
    h = w
    y = int(y - h / 2)
    # print(y, h, x, w)
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=24, inter=cv2.INTER_CUBIC)

    return roi


if __name__ == "__main__":
    # Define counter
    COUNTER = 0
    ALARM_ON = False
    MAX_FRAME = 20
    # Load model
    model = loadModel('trained_model/model_keep.json', "trained_model/weight_keep.h5") #Model Loaded

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()

    # loop over frames from the video stream
    counter = 0
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels, and calculate pupil ratio)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # detect faces in the grayscale frame
        rects = predictFacialLandmark(img=frame, detector=detector)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('temp/frame/frame{0}.png'.format(counter), frame)
        # loop over the face detections
        for rect in rects: # 0 - False, 1 - True
            counter += 1
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEyeRatio = shape[lStart:lEnd]
            leftEye = getRoi(leftEyeRatio, frame)
            
            # cv2.imwrite('images/left/left{0}.png'.format(counter), leftEye)
            classLeft = predictImage(leftEye, model=model)

            rightEyeRatio = shape[rStart:rEnd]
            rightEye = getRoi(rightEyeRatio, frame)
            # cv2.imwrite('images/right/right{0}.png'.format(counter), rightEye)
            classRight = predictImage(rightEye, model=model)

            leftEyeHull = cv2.convexHull(leftEyeRatio)
            rightEyeHull = cv2.convexHull(rightEyeRatio)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # print(classLeft, classRight)

            if classLeft == 0 and classRight == 0: #Both eye 0 then counter keeps on adding
                COUNTER += 1
                
                cv2.imwrite('dataset/closed_eyes/left{0}.png'.format(counter), leftEye) # New images in folder (closed)
                cv2.imwrite('dataset/closed_eyes/right{0}.png'.format(counter), rightEye)

                cv2.putText(frame, "Closing", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "NF: {:.2f}".format(COUNTER), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= MAX_FRAME:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON: #IF already on make this makes it dosent go on off again and again)
                        ALARM_ON = True
                        playsound('alarm.wav')
                        print('Drowsy')
                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSY!", (100, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        COUNTER = 0
                        ALARM_ON = False
            else:
                COUNTER = 0
                ALARM_ON = False
                cv2.imwrite('dataset/open_eyes/left{0}.png'.format(counter), leftEye) # New images in folder (open)
                cv2.imwrite('dataset/open_eyes/right{0}.png'.format(counter), rightEye)
                cv2.putText(frame, "Opening", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # show the frame

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
