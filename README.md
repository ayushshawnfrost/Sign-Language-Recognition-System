# Sign Language Recognition System

## Description
Sign language recognition with deep learning techniques can contribute to equity and benefit the
hearing-impaired community, especially infants and children. This proposal is to complete an Isolated
Sign language recognition task with deep learning models using a publicly-available the Isolated
Sign Language Recognition corpus (version 1.0) dataset available on Kaggle and use this to raise
awareness of Sign Language learning. This project is aimed to recognize and interpret sign language
gestures to spoken english language. This recognition technology, hosted on an accessible
web application, which can be used to empower hearing parents with the means to, ensuring that
deaf infants are not deprived of vital language acquisition opportunities during their formative years.
Thus, this project will have two major sections: first, development and training of Deep
Learning model for sign language classification; second, creating a user-friendly web application
utilizing the classification model to promote sign language awareness.

## Technologies Used
- Tensorflowjs
- ReactJS
- Mediapipe Hollistic Solution

## Methodology
After a brief literature review of current available solutions for classification task in general, we came to a conclusion
that our project should have four phase. First: Exploratory Data Analysis for the Isolated Sign Language Recognition
corpus Dataset. Second: developing a model that can classify American sign language given some image/landmark.
Third: a web application, mounting the sign recognition model, making the system more accessable. Forth: Deployment
of the model and the web application on the internet.

### Dataset
The dataset is a publicly available dataset on Kaggle. The Isolated Sign Language Recognition corpus (version 1.0)
[6] is a collection of hand and facial landmarks generated by Mediapipe version 0.9.0.1 on 100k videos of isolated
signs performed by 21 Deaf signers from a 250-sign vocabulary (e.g. “mom", “dad", “help") which represents the first
concepts taught to infants in any language. Some of the 250 signs can be seen on Figure 1. The dataset was created by
Deaf Professional Arts Networks and the Georgia Institute of Technology. 21 signers recruited by the Deaf Professional
Arts Network provided the sign. They are from many regions across the United States and all use American Sign
Language as their primary form of communication. They represent a mix of skin tones and genders [6]. Each video
was annotated at creation time by the smartphone app. Videos were coarsely reviewed by researchers at the Georgia
Institute of Technology to attempt to remove poor recordings. The input modality for the dataset is vision-based using
hand gestures, facial expressions, and posture.


![image](https://github.com/ayushshawnfrost/Sign-Language-Recognition-System/assets/23500476/0e8a5875-5332-4b77-99f4-1f62ea9c05dc)


### Model for Classification
After a brief literature review of current available solutions for classification task in general, we came to a conclusion
that our project should have two parts, first: developing a model that can classify American sign language given some
image/landmark and second: a program that can utilizes web cam and transfer the sequence of frames to the model for
prediction
In the first phase of this project, we have implemented a deep learning model to perform a classification task for these
250 signs. Starting from a Google-provided baseline model, it is intended to develop and apply more complex models
to achieve baseline accuracy of greater than 60 percent.
We harnessed the modularity of custom Keras layers to conduct preprocessing operations on 3D landmarks and facilitate
feature extraction. Our approach involved the application of distinct layers, each with a specific purpose. One layer
was designed to isolate the appropriate hand landmarks, while another was employed to extract landmarks from the
upper and lower lips. We also developed a layer to compute the Euclidean distance between non-adjacent joints of
the signing hand, as well as between non-adjacent joints of the upper and lower lips. Furthermore, we incorporated a
layer to calculate the angle between the x and y vectors formed by non-adjacent joints of the hand. This comprehensive
approach allowed us to effectively process and analyze the sign language data.After modification and training on the
Kaggle data set, the training accuracy of the underlying model came out to be 62 percent.

Notice that the focus of this study right now is on building a working model and then deploying it in an application
rather than achieving a best classification model. There is a chance to improve this model by various other techniques
like considering a different architecture like Convolutional Neural Networks (CNN)[7] or Long Short-Term Memory
(LSTM). Also, there is a possibility of improvement by Hyperparameter Tuning, considering different loss functions and
many more. In a word, we studied the state-of-the-art researches on isolated sign language recognition and proposed
our own model.
To make this sign language recognition model accessable to over the internet, we have hosted it on AWS S3 storage
bucket. This kind of deployment will help the our web application to load the model on client side by downloading the
model from S3 bucket.


![image](https://github.com/ayushshawnfrost/Sign-Language-Recognition-System/assets/23500476/7b03e3cc-56b0-47ff-864c-6683a49bbf3f)

For the second part, we developed a supporting program which extracts the landmarks from a live webcam feed. This
is achieved with the help of OpenCV library and Google’s Mediapipe Holistic Solution. OpenCV is a library of
programming functions mainly for real-time computer vision. MediaPipe Holistic is a comprehensive solution for
human body landmark detection developed by Google. It integrates pose, face, and hand landmarkers to create a
complete landmarker for the human body. This allows for the analysis of full-body gestures, poses, and actions. The
solution applies a machine learning model to a continuous stream of images and outputs a total of 543 landmarks in
real-time. These landmarks include 33 pose landmarks, 468 face landmarks, and 21 hand landmarks for each hand. An
example of this prediction can be seen on Figure 2.
The MediaPipe Holistic solution is highly optimized, enabling simultaneous detection of body and hand pose and
face landmarks on mobile devices. It allows for the interchangeability of the three components (pose, face, and hand
landmarkers), depending on the quality/speed trade-offs. This holistic approach provides a more complete and detailed

understanding of human body language and movement, making it particularly useful in applications such as sign
language recognition, fitness tracking, gaming, and more.

### Web application
The developed sign language recognition model’s direct usability will be limited to those with proficiency in machine
learning techniques. To broaden its accessibility and make it usable even for individuals with basic web navigation
skills, the model is integrated into a web application. By doing so, it’s reach can be significantly expanded, benefiting
both newcomers and experienced users.
The website requires a webcam to capture the live feed and a speaker optionaly. It has a section which includes
prerecorded videos teaching how to do some common American sign language. These short videos can assist a user to
learn a sign and try it out in the "Try Signing Below" section. The practice section of the web application can be seen on
Figure 3. The user can see themselves on the web-app while signing. The web-application also has the ability to dictate
the sign prediction out loud through speaker. A list of tips on how to get an accurate prediction is also mentioned on the
web application. The prediction section of the web application can be seen on Figure 4.


![image](https://github.com/ayushshawnfrost/Sign-Language-Recognition-System/assets/23500476/6b1949f7-e549-44e1-8eb0-99ae9f1c6405)


We have utilized React.js which is a very powerful javaScript library to develop the front-end. To utilize our sign
language recognition model, we used tensorflow.js. TensorFlow.js is an open-source hardware-accelerated JavaScript
library for training and deploying machine learning models. The sign recognition model requires the landmarks for
face, pose, left hand and right hand as a input for prediction. This landmarks are extracted by using a yet another model
called Mediapipe hollistic solution by Google. These landmarks are fed into the sign language recognition model to get
a prediction.

![image](https://github.com/ayushshawnfrost/Sign-Language-Recognition-System/assets/23500476/bcaa7496-38c1-4dc0-acdb-1038fdf609f1)


Furthermore, we also plan to release this application as an open-sourced project. Engaging with the open-source
community has the power to amplify the project’s influence, making it accessible to a wider array of audiences and
sectors. Collaborative efforts could lead to greater impact and adoption across various domains.


###  Deployment Strategy
Deployment of model on the web is very crucial process for this project. After analysing the deployment process and
architecture for common machine learning models to the web, I came to understand that there can be multiple ways of
deployment depending on various factor like size of model, format of models and end device’s computing capacity[
1]. First and common approach for the deployment of machine learning models on the web involves developing a
Restful API [2]. In this approach a Restful API is developed generally using Fast API or Django and then the model
is wrapped inside the Restful API. Clients can utilise the model using common http protocol i.e GET,POST. In case
the input for model is an image or a string, passing that in the API call and getting result from the model as response
from the API. This means that the all the processing is done at the server-side where API is deployed. Hence, any
size model can deployed using this approach and it also does not depend much upon the type of framework/library
used to develop the model. However, a point to be considered utilizing this kind of approach for our project is that,
should we send a video as payload with the API call? or do an API calls for each frame of a video as payload? what
is the maximum duration for the video to get processes easily and efficiently? How can we archive a real time sign
detection? Second approach arises from the idea of performing the computation at the client side rather then server side.
This ideology is discussed in detail in [3]. TensorFlow.js is a library for machine learning in JavaScript. It supports
developing ML models in JavaScript, and using ML directly in the browser or in Node.js. In this approach the final
model after training is generally in json format. This Json format model is then typically hosted in a cloud storage
(e.g. AWS 2 S3) and loaded in the browser by downloading that model from cloud. Once the model is loaded, all the
6
prediction can be done in the browser itself using JavaScript. Real time detection can be easily done because there is no
need to send frames/videos to and from server/API. However, there are certain limitations of using this kind of approach.
TensorFlow.js is limited to smaller models due to the performance limitations of JavaScript engines. JavaScript engines
are less powerful compared to specialized machine learning frameworks and hardware accelerators such as GPUs or
TPUs.Tensorflow.js has far less libraries compared to TensorFlow Python means not all operations can be performed
the same way as TensorFlow.


## Installation
1. Clone the repository
2. Navigate to the project directory
3. Install the dependencies with `npm install`
4. Start the application with `npm start`


