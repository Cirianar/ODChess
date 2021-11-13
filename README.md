# ODChess
The github repository "Object-Detection Chess" is for an app which utilises an object detection model trained with custom data to locate and identify chess pieces on a chess board. The github repository includes a python executable application enabling the usage of this model.

Credits to *YOLOv5 🚀 by Ultralytics (https://github.com/ultralytics/yolov5), GPL-3.0 license* for providing source code in the detection.py file

__Download and install steps__
  1) Download this repository as a ZIP file and extract its content
  2) Download and install Anaconda
  3) Download and install Python3
  4) Open a new Anaconda prompt and create a virtual environment by using the command `conda create --name myenv`
  5) Activate the new environment by using the command `activate myenv`
  6) Install all following repositories by using the command `pip install {module name}`
      - opencv-python
      - torch
      - pandas
      - requests
      - pillow
      - yq
      - tqdm
      - torchvision
      - matplotlib
      - seaborn
  7) Download and install the YOLOV5 repository at https://github.com/ultralytics/yolov5 
  8) Navigate inside the directory of the ODChess folder by using the command cd path/to/ODChess
  9) Copy the path to the YOLOV5 directory, which was installed in step 5, on to the clipboard
  10) Use the following command to activate the object detection of chess pieces:
      `python ODChess.py --yolov5 path\\to\\yolov5_directory --source 0 --weights C:\Users\aidan\Desktop\ODChess\data\models\model_yolo5_v1\checkpoint\best.pt`
      *Make sure to replace each “\” in the path to the YOLOV5 directory with a double “\\” when pasting it into the command*

