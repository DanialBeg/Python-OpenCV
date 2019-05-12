# Importing all packages needed for program
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# Parsing the arguments together
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# Load our input image
image = cv2.imread(args["image"])
orig = image.copy()

# Store image dimensions
(H, W) = image.shape[:2]

# Set new width and height and find the ratio of change
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# Resizing the image
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Defining layernames for EAST model, first layer gives probability and second gives box coordinates
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# Load EAST model
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# Making a blob and then doing a forward pass to obtain our layers
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# How long it took to predict the text
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# Store the rows and columns from scores then setup our rectangle and confidences lists
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop through the rows
for y in range(0, numRows):
	# grab the probabilities and then data for our rectangles
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# Loop through the columns from scores
	for x in range(0, numCols):
		# ignore scores with low probabilities
		if scoresData[x] < args["min_confidence"]:
			continue

		# compute offset
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# get rotation angle and then its sin and cos
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# find width and height of the box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# find coords for the box using the data from above
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add above calculations to our rectangles and confidence lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# Supress weak overlapping boxes in final output
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop through our bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# draw the bounding boxes
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Show output
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
