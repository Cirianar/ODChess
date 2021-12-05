import tkinter as tk

from tkinter.constants import BOTTOM, LEFT, RIGHT
from PIL import Image, ImageTk

class Window:
    def __init__(self):
        self.windowTitle = "ODChess Stats"

        # Creates a window
        self.window = tk.Tk()
        self.window.title(self.windowTitle)

        self.blackLabels = tk.Frame(self.window)
        self.whiteLabels = tk.Frame(self.window)
        self.stream = tk.Frame(self.window)

        self.blackLabels.pack(side=LEFT, padx=20, pady=20)
        self.whiteLabels.pack(side=RIGHT, padx=20, pady=20)
        self.stream.pack(side=BOTTOM)

        self.sections = (self.blackLabels, self.whiteLabels)

    def getDetectionData(self, detections, names):
        detectionsList = []

        for n in names: 
            detectionNum = 0 # Number of times object is detected
            detectionCat = "black" if n[:5] == "black" else "white"

            for i in detections[:, -1].unique(): # All elements which are the same are grouped and itterated over
                if names[int(i)] == n:
                    detectionNum = (detections[:, -1] == i).sum()  # detections per class
            
            element = (n, detectionCat, detectionNum)
            detectionsList.append(element)

        # Splits the detection list in two
        black = detectionsList[:len(detectionsList)//2]
        white = detectionsList[len(detectionsList)//2:]

        detectionsList = (black, white)

        return detectionsList

    def generateTitles(self):
        for list in self.sections:
            category = "black" if list == self.blackLabels else "white"

            displayText = f"{category} chess pieces"
            displayText = displayText.title()

            header = tk.Label(list, text=displayText, font="Verdana 10 bold")
            header.pack()
    
    def generateLabels(self, names):
        self.generateTitles()

        for i, names in enumerate(names):
            nameCat = names[:5]
            list = self.blackLabels if nameCat == "black" else self.whiteLabels

            label = tk.Label(list, text="ERROR: NO DATA")
            label.pack()
    
    def generateStream(self):
        label = tk.Label(self.stream, image=None)
        label.pack()

    def updateLabels(self, detections, names):
        detectionsList = self.getDetectionData(detections, names)

        for i, category in enumerate(detectionsList):
            for i, labels in enumerate(self.sections[i].winfo_children()):
                i = i-1 # Skips the first child since thats the header
                if i != -1:
                    name = category[i][0]
                    name = name.replace("-", " ")
                    name = name.title()
                    count = category[i][2]
                    labels['text'] = f"{name}: {count}"

    def updateStream(self, resultImg):
        stream_widgets = self.stream.winfo_children()
        stream_label = stream_widgets[0]

        img =  ImageTk.PhotoImage(image=Image.fromarray(resultImg))

        stream_label['image'] = img

    def buildWidgets(self, names):
        self.generateLabels(names)
        self.generateStream()

    def updateWindow(self, detections, names, resultImg):
        self.updateLabels(detections, names)
        self.updateStream(resultImg)
        # Function to update the window
        self.window.update()