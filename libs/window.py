import tkinter as tk

class Window:
    def __init__(self):
        # Creates a window
        self.window = tk.Tk()

    def configureTopbar(self, windowTitle):
        # Sets the title of the window to the 1st position argument
        self.window.title(windowTitle)

    def addButton(self, buttonCommand, buttonName):
        # Returns a button with buttonCommand as function and buttonText as display name
        return tk.Button(self.window,command=buttonCommand,text=buttonName).pack()

    def addLabel(self):
        # Returns a label
        return tk.Label(self.window).pack()

    def configureLabel(self, labelName, image):
        # Used to hold a video feed in a label element
        # The 1st positional argument refers to the name of the label one wants to configure
        # The 2nd positional argument refers to the image, should be COLOR_BGR2RGBA format
        labelName.imgtk = image
        labelName.configure(image=image)
        labelName.after(10, self.configureLabel(labelName, image)) # Creates a self referencial loop

    def build(self):
        # Function to build the window
        self.window.mainloop()