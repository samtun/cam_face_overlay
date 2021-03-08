import pyvirtualcam
import cv2

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
overlayImage = cv2.imread("images/smile_glasses.png", -1)

video = cv2.VideoCapture(0)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

facePadding = 60

with pyvirtualcam.Camera(width=width, height=height, fps=fps, print_fps=20) as cam:
    print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')
    count = 0
    while True:
        # Restart video on last frame.
        if count == length:
            count = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read video frame.
        ret, frame = video.read()
        if not ret:
            raise RuntimeError('Error fetching frame')

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
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
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            resized_image = cv2.resize(overlayImage, (w, h)) 
            alpha_s = resized_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            y2 = y+h
            x2 = x+w
            if x > 0 and y > 0 and x2 < width and y2 < height:
                for c in range(0, 3):
                    frame[y:y2, x:x2, c] = (alpha_s * resized_image[:, :, c] + alpha_l * frame[y:y2, x:x2, c])
        
        # Send to virtual cam.
        cam.send(frame)
        cv2.imshow("Faces found", frame)

        # Wait until it's time for the next frame
        cam.sleep_until_next_frame()
        count += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
