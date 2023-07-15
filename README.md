# Face Detection using CNN (MobileNetV2)

## About the project
+ This project uses **MobileNetV2** pretrained model for face **detection** and **localization** (implemented in **Tensorflow**).
+ The model outputs two things, one is the **probability** of **detecting a face**, and second the **bounding box coordinates** for the face.

## Dataset
+ The dataset was locally created, images were captured **manually** and labelled using `labelme` library.
+ To cover up with less amount of data, **data augmentation** pipeline was created using `Albumentations` library.

## Tools
+ Numpy
+ Matplotlib
+ Tensorflow
+ OpenCV
+ Labelme ([open-source](https://github.com/wkentaro/labelme))
+ Albumentations ([site link](https://albumentations.ai/))
+ MobileNetV2 ([article](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c))

## Model, Architecture and Loss Functions
+ MobileNet architecture consists of two blocks. (_refer article_)
![mobilenetv2](helpers\mobilenetv2architecture.jpg)
+ Since it is **localization** model, it has two tasks namely **classification** and **regression**.
+ Classification deals with **probability of detecting a face**, and Regression deals with **estimating bounding box coordinates**.
+ **Binary crossentropy** loss was used for classification and **Mean squared error** for regression. A **weighted sum** of these losses was taken to define the total loss.

## Performance
+ The model performed quite well considering the availibility of data.
+ Below are the plot for training
![train_plot](helpers\plot.png)
+ As observed, further training was stopped to avoid overfitting to train data.

## Result
![face_detected](helpers\Screenshot(319).png)

## Running locally
+ clone this repo in local machine.
+ create and switch to virtual environment.
+ install the dependencies from `requirements.txt`. (`pip install -r requirements.txt`)
+ run the script `testing.py`.
+ if one wants to exit, click the openCV window and press the `Q` key on keyboard.
