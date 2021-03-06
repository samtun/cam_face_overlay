import cv2
import sys

cascPath = sys.argv[1]
overlayImagePath = sys.argv[2]
overlayImage = cv2.imread(overlayImagePath, -1)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
im_width, im_height = (cap.get(3), cap.get(4))

facePadding = 60

while True:
    ret, image_bgr = cap.read()

    if not ret:
        print("failed to grab frame")
        break

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        image_bgr,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(200, 200),
        flags = cv2.cv2.IMREAD_REDUCED_GRAYSCALE_2
    )

    cv2.namedWindow('Detenct Me Senpai', cv2.WINDOW_NORMAL)

    for (x, y, w, h) in faces:
        x = x - facePadding
        y = y - facePadding
        w = w + facePadding * 2
        h = h + facePadding * 2
        resized_image = cv2.resize(overlayImage, (w, h)) 
        # Draw a rectangle around the faces
        # cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        alpha_s = resized_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        y2 = y+h
        x2 = x+w
        for c in range(0, 3):
            image_bgr[y:y2, x:x2, c] = (alpha_s * resized_image[:, :, c] + alpha_l * image_bgr[y:y2, x:x2, c])

    cv2.imshow("Faces found", image_bgr)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break