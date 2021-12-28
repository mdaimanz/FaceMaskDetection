# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from djitellopy import tello
import math

# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamon()
# me.takeoff()
# me.send_rc_control(0,0,25,0)
# time.sleep(2.2)

w, h = 300,400
fbRange = [15000, 22000]
pid = [0.4, 0.4, 0] ### [proportional, integral ,derivatives]
pError =0

def trackFace(area, center, w, pid, pError):
	x , y = center
	forward_backward = 0
	error = x - w // 2
	speed = pid[0] * error + pid[1] * (error - pError)  # utk kira yaw
	speed = int(np.clip(speed, -100, 100))

	if area > fbRange[0] and area < fbRange[1]:
		forward_backward = 0
	elif area > fbRange[1]:
		forward_backward = -20
	elif area < fbRange[0] and area != 0:
		forward_backward = 20

	# print(speed, forward_backward)
	if x == 0:
		speed = 0
		error = 0
	print("Area: ", area)
	print("RC Control: ", 0,forward_backward,0,speed)
	# me.send_rc_control(0, forward_backward, 0, speed)
	return error


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	myFaceListArea = []
	myFaceListCenter = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			myW = endX - startX
			myH = endY - startY
			centerX = (myW // 2) + startX
			centerY = (myH // 2) + startY
			area = myW * myH
			myFaceListArea.append(area)
			myFaceListCenter.append([centerX, centerY])

			faces.append(face)
			locs.append((startX, startY, endX, endY))



	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)


	# return a 2-tuple of the face locations and their corresponding
	# locations

	##img, [myFaceListCenter[i], myFaceListArea[i]]
	if len(myFaceListArea) != 0:
		i = myFaceListArea.index(max(myFaceListArea))
		return (locs, preds, myFaceListCenter[i], myFaceListArea[i])
	else:
		return (locs, preds, [0, 0], 0)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds, center, area) = detect_and_predict_mask(frame, faceNet, maskNet)
	pError = trackFace(area, center, w, pid, pError)
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		width = endX - startX
		height = endY - startY
		cX = (width // 2) + startX
		cY = (height // 2) + startY
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"

		if label == "No Mask":
			# capture image if wear no mask
			cv2.imwrite(f'Resources/Images/{time.time()}.jpg', frame)
			time.sleep(0.3)
			print("Image captured")

		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, (0, 255, 0), cv2.FILLED)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		##me.land()
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()