# Breast_Cancer_Detection_ML

![img](https://github.com/0-0Dibakar/Breast_Cancer_Detection_ML/assets/106139442/e9262ec5-685d-4f65-8730-63de8c197245)

In this project in python, we’ll build a classifier to train on 80% of a breast cancer histology image dataset. Of this, we’ll keep 10% of the data for validation. Using Keras, we’ll define a **CNN (Convolutional Neural Network)**, call it CancerNet, and train it on our images. We’ll then derive a confusion matrix to analyze the performance of the model.

IDC is Invasive Ductal Carcinoma; cancer that develops in a milk duct and invades the fibrous or fatty breast tissue outside the duct; it is the most common form of breast cancer forming 80% of all breast cancer diagnoses. And histology is the study of the microscopic structure of tissues.

**Breast Cancer Classification – Objective**

To build a breast cancer classifier on an IDC dataset that can accurately classify a histology image as benign or malignant.
**What is Deep Learning?**

An intensive approach to Machine Learning, Deep Learning is inspired by the workings of the human brain and its biological neural networks. Architectures as deep neural networks, recurrent neural networks, convolutional neural networks, and deep belief networks are made of multiple layers for the data to pass through before finally producing the output. Deep Learning serves to improve AI and make many of its applications possible; it is applied to many such fields of computer vision, speech recognition, natural language processing, audio recognition, and drug design.

**What is Keras?**

Keras is an open-source neural-network library written in Python. It is a high-level API and can run on top of TensorFlow, CNTK, and Theano. Keras is all about enabling fast experimentation and prototyping while running seamlessly on CPU and GPU. It is user-friendly, modular, and extensible.

**Steps for Advanced Project in Python – Breast Cancer Classification**

1. Download this zip. Unzip it at your preferred location, get there
2. Now, inside the inner breast-cancer-classification directory, create directory datasets- inside this, create directory original:
3. Download the dataset.
4. Unzip the dataset in the original directory. To observe the structure of this directory, we’ll use the tree command:
   We have a directory for each patient ID. And in each such directory, we have the 0 and 1 directories for images with benign and malignant content.
   
**config.py:**
Here, we declare the path to the input dataset (datasets/original), that for the new directory (datasets/idc), and the paths for the training, validation, and testing directories using the base path. We also declare that 80% of the entire dataset will be used for training, and of that, 10% will be used for validation.
This holds some configuration we’ll need for building the dataset and training the model. You’ll find this in the cancernet directory.   

**build_dataset.py:**

This will split our dataset into training, validation, and testing sets in the ratio mentioned above- 80% for training (of that, 10% for validation) and 20% for testing. With the ImageDataGenerator from Keras, we will extract batches of images to avoid making space for the entire dataset in memory at once.

In this, we’ll import from config, imutils, random, shutil, and os. We’ll build a list of original paths to the images, then shuffle the list. Then, we calculate an index by multiplying the length of this list by 0.8 so we can slice this list to get sublists for the training and testing datasets. Next, we further calculate an index saving 10% of the list for the training dataset for validation and keeping the rest for training itself.

Now, datasets is a list with tuples for information about the training, validation, and testing sets. These hold the paths and the base path for each. For each setType, path, and base path in this list, we’ll print, say, ‘Building testing set’. If the base path does not exist, we’ll create the directory. And for each path in originalPaths, we’ll extract the filename and the class label. We’ll build the path to the label directory(0 or 1)- if it doesn’t exist yet, we’ll explicitly create this directory. Now, we’ll build the path to the resulting image and copy the image here- where it belongs.

5. Run the script **build_dataset.py:**

       py build_dataset.py
   
**cancernet.py:**

The network we’ll build will be a CNN (Convolutional Neural Network) and call it CancerNet. This network performs the following operations:

Use 3×3 CONV filters
Stack these filters on top of each other
Perform max-pooling
Use depthwise separable convolution (more efficient, takes up less memory)

We use the Sequential API to build CancerNet and SeparableConv2D to implement depthwise convolutions. The class CancerNet has a static method build that takes four parameters- width and height of the image, its depth (the number of color channels in each image), and the number of classes the network will predict between, which, for us, is 2 (0 and 1).

In this method, we initialize model and shape. When using channels_first, we update the shape and the channel dimension.

Now, we’ll define three DEPTHWISE_CONV => RELU => POOL layers; each with a higher stacking and a greater number of filters. The softmax classifier outputs prediction percentages for each class. In the end, we return the model.

**train_model.py:**

In this script, first, we set initial values for the number of epochs, the learning rate, and the batch size. We’ll get the number of paths in the three directories for training, validation, and testing. Then, we’ll get the class weight for the training data so we can deal with the imbalance.

Now, we initialize the training data augmentation object. This is a process of regularization that helps generalize the model. This is where we slightly modify the training examples to avoid the need for more training data. We’ll initialize the validation and testing data augmentation objects.

We’ll initialize the training, validation, and testing generators so they can generate batches of images of size batch_size. Then, we’ll initialize the model using the Adagrad optimizer and compile it with a binary_crossentropy loss function. Now, to fit the model, we make a call to fit_generator().

We have successfully trained our model. Now, let’s evaluate the model on our testing data. We’ll reset the generator and make predictions on the data. Then, for images from the testing set, we get the indices of the labels with the corresponding largest predicted probability. And we’ll display a classification report.

Now, we’ll compute the confusion matrix and get the raw accuracy, specificity, and sensitivity, and display all values. Finally, we’ll plot the training loss and accuracy.
This trains and evaluates our model. Here, we’ll import from keras, sklearn, cancernet, config, imutils, matplotlib, numpy, and os.

**Output**

![img](https://github.com/0-0Dibakar/Breast_Cancer_Detection_ML/assets/106139442/25ddb829-9e8c-4ca5-9ff3-b9110caea9af)


**Summary**

In this project in python, we learned to build a breast cancer classifier on the IDC dataset (with histology images for Invasive Ductal Carcinoma) and created the network CancerNet for the same. We used Keras to implement the same. Hope you enjoyed this Python project.
