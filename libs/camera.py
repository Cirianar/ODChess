import cv2 as cv

from PIL import Image,ImageTk

class Camera:
    def __init__(self, frameWidth, frameHeight):
        # Defines the width and height of the label in terms of px
        self.width = frameWidth # 1920px
        self.height = frameHeight # 1080px

        self.stream = cv.VideoCapture(0)

        # Sets the stream wdith and height
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
    
    def readFrame(self):
        # Reads 1 frame from the webcam
        ret, frame = self.stream.read()

        # Flips the image along the Y axis
        frame = cv.flip(frame, 1)
        # Converts frame to COLOR_BGR2RGBA
        frame_image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        # Converts image to PIL format
        img = Image.fromarray(frame_image)

        # This makes the image TKinter compatiable
        # TKinter is the module which creates the window
        imgtk = ImageTk.PhotoImage(image=img)

        # Returns the final image to be used
        return imgtk





