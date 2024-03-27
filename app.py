import cv2
import face_recognition


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
known_image = face_recognition.load_image_file("opencv_frame_0.png")

OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


def compareFace(known_encoding, result_encoding):
    return face_recognition.compare_faces(
        [known_encoding], result_encoding)


def encodeImage(image):
    return face_recognition.face_encodings(image)


def detectLargestFace():
    capture = cv2.VideoCapture(0)

    # Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    # Start the window thread for the two windows we are using
    cv2.startWindowThread()

    rectangleColor = (0, 165, 255)

    try:
        while True:
            rc, fullSizeBaseImage = capture.read()

            baseImage = cv2.resize(fullSizeBaseImage, (320, 240))

            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                cv2.destroyAllWindows()
                exit(0)

            resultImage = baseImage.copy()
            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            # Encoding camera face
            result_encoding = []
            if encodeImage(resultImage):
                result_encoding = encodeImage(resultImage)[
                    0]
            known_encoding = encodeImage(known_image)[0]

            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0

            for (_x, _y, _w, _h) in faces:
                if _w*_h > maxArea:
                    x = _x
                    y = _y
                    w = _w
                    h = _h
                    maxArea = w*h
            if maxArea > 0:
                cv2.rectangle(resultImage,  (x-10, y-20),
                              (x + w+10, y + h+20),
                              rectangleColor, 2)
                if any(result_encoding):
                    result = compareFace(known_encoding, result_encoding)
                    print(result)

            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)

    except KeyboardInterrupt as e:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == '__main__':
    detectLargestFace()
