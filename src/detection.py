import cv2
import sys

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
overlayImage = cv2.imread("images/smile_glasses.png", -1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
fps = video.get(cv2.CAP_PROP_FPS)

im_width, im_height = (cap.get(3), cap.get(4))

facePadding = 60

with pyvirtualcam.Camera(width=1280, height=720, fps=fps, print_fps=20) as cam:
    print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')
    count = 0
    while True:
        
        # Restart video on last frame.
        if count == length:
            count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, image_bgr = cap.read()
        height, width, channels = image_bgr.shap

        if not ret:
            print("failed to grab frame")
            break

        # Convert to RGB.
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            image_bgr,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(100, 100),
            flags = cv2.cv2.IMREAD_REDUCED_GRAYSCALE_2
        )

        for (x, y, w, h) in faces:
            x = x - facePadding
            y = y - facePadding
            w = w + facePadding * 2
            h = h + facePadding * 2
            # Draw a rectangle around the faces
            # cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            resized_image = cv2.resize(overlayImage, (w, h)) 
            alpha_s = resized_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            y2 = y+h
            x2 = x+w
            if x > 0 and y > 0 and x2 < width and y2 < height:
                for c in range(0, 3):
                    image_bgr[y:y2, x:x2, c] = (alpha_s * resized_image[:, :, c] + alpha_l * image_bgr[y:y2, x:x2, c])
        
        # Send to virtual cam.
        cam.send(image_bgr)

        # Wait until it's time for the next frame
        cam.sleep_until_next_frame()
        count += 1

        cv2.imshow("Faces found", image_bgr)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break