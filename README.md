# Generate Caption from image using neural network

## Introduction <br>

As social media gets intertwined in our day to day life more and more, every day we see our family and friends pushing their images on social media. This dataset can be used to recognize  the place and object in the images. As a aspiring Data science student at BrainStation, I was curious to discover that how can I identify objects and places in a given image. <br>

Inspired by the [Ph.D. thesis](https://cs.stanford.edu/people/karpathy/main.pdf) of Andrej Karpathy (Director of AI and Autopilot Vision at Tesla), I have tried to train a model that can generate a caption for an image using neural networks.<br>

As I am making efforts to transition into a career in the Data Science space, this is the first large independent project that I have taken on as my Capstone project for the course. I thoroughly enjoyed the entire journey of the project design and development from the research phase to the delivery phase.

## Dataset <br>

For model training purpose, I have used [MS COCO](https://cocodataset.org/#download) data set which has more than 118k training images, 5k validation images and 40 k test images. For train and validation set, there are 5 related captions for each images. The benefit of using such a large data set is that, the model will be trained on huge variety of objects, colours and actions, which will help the model to predict the better captions.<br>

## Data Cleaning <br>

To clean the data, some basic cleaning steps was required. Such as, the image ID in data folder (or in the URLs) is in between 2 digit to 12 digit in length. To make the image ID in uniform format, I performed the zero padding to make all the IDs 12-digits in length, so that I can access the image file later in the code by prefixing folder path and suffixing file extension (.jpg) to the 12-digit image id. To bring the uniformity in the words of caption, I decided to lower case all the words. I had to make the collection of all the unique words in the caption corpus. In order to do this, I converted all the words in lowercase, so that I don’t confuse the model between “A” and “a” as two different words. Remove all the punctuations, special character, digits in caption, multiple white spaces in between the words.<br>

## Data Preparation <br>

Following steps are required before training the model:<br>
1. Make a list of all the five related captions
2. Add "start" and "end" sequence in each caption
3. Extract the image feature vector to train the model
4. Make a list of all the unique words in corpus
5. Define a maximum length of predicted caption
6. Prepare a fixed length vector for all the unique words

For modelling purposes, two models are required, one to extract feature vectors from images and one for predicting the sequence of words for captions.

To process the images, I have used **InceptionV3** model (to extract the image feature vector) i.e. Convolutional Neural Network (CNN) with weights of "ImageNet", where I removed the "softmax" layer (which is 1000 class classification of images). To process the words of captions, I used GloVe model (GloVe 300-Dimensional word vectors trained on Wikipedia and Gigaword 5 data) to map every word in the vocabulary of the caption corpus to a vector of finite length. The matrix is nothing but the 300 length numerical presentation of all the unique words in captions, that are being used in formation of cations for 118k images.<br>

## Model Architecture <br>

To train the model, pass the image feature vector (extracted using CNN model) and sequence of partial caption to Recurrent Neural Network (RNN-LSTM), the model will find the next possible word from the word matrix with the highest probability to create the next sequence of partial caption. Partial caption for the first time will be the "<start>" sequence. This can better be understood using following diagram:<br>

<img src="https://github.com/raivarsha14/ImageCap/blob/master/images/Screen%20Shot%202020-07-02%20at%207.11.27%20PM.png"
     alt="Model Architecture"
     style="float: left; margin-right: 10px;" />

For more clear understanding, we can say that in the following diagram, if we pass the image feature vector along with the partial caption as "<start> a woman is", the partial caption will pass through the RNN model. The image feature vector and output of RNN (from the previous step) will pass through feed forward neural network & model will search for the next possible word for the caption with the highest probability in word corpus and will generate the next sequence of partial caption.<br>

<img src="https://github.com/raivarsha14/ImageCap/blob/master/images/Screen%20Shot%202020-07-02%20at%207.11.38%20PM.png"
     alt="Model Architecture"
     style="float: left; margin-right: 10px;" />

The model does this process until it either reaches the maximum caption length (as defined in the data preparation step) or finds the "<end>" sequence of the caption.<br>

<img src="https://github.com/raivarsha14/ImageCap/blob/master/images/Screen%20Shot%202020-07-02%20at%207.11.47%20PM.png"
     alt="Model Architecture"
     style="float: left; margin-right: 10px;" />

The model structure will be like this:<br>

<img src="https://github.com/raivarsha14/ImageCap/blob/master/images/download.png"
     alt="Model Architecture"
     style="float: left; margin-right: 10px;" />

The model structure shows that the maximum length of cation can be 52 words, 384 neurones are being used, 300 is the length of word vector and 2048 is the length of image vector. The model will generate the probability for 7068 unique words in corpus.<br>

## Caption Prediction <br>

To predict the caption, two methods are shown. Agrmax and Beam search. Argmax is a $greedy search$ method. Agrmax function will predict the caption from the words with the highest probability in the word corpus. It ignores the global maxima, which refers to largest output. In contrast, Beam Search is a heuristic search algorithm which expands upon the greedy search. It selects multiple alternatives for an input sequence at each time step based on conditional probability. The number of multiple alternatives depends on a parameter called Beam Width. This works by taking multiple combinations of words (in our case) for consecutive time steps and calculating which words are likely to go together based on the conditional probability. An upside of higher beam width is that it should yield better result as the multiple candidate sequences increase the likelihood of better matching a target sequence. The downside of higher beam width is that it would use a lot of memory and computational power.

## Model Evaluation <br>

To evaluate the model, I applied the BLEU (Bilingual Evaluation Understudy) score that was initially proposed by Kishore Papineni in his [research paper](https://www.aclweb.org/anthology/P02-1040.pdf). BLEU compares the n-gram of the generated caption with the n-gram of the original caption to count the number of matches. These matches are independent of the positions where they occur. It lies between 0 to 1. 1 means the exact same match whereas 0 means no similar words at all. Even with the synonyms, the score tends to get down. For example<br>
Actual cap: “A small dog is running on green grass”<br>
Predicted cap: “A small dog is running on green grass”<br>
BLEU score will be 1 because of the exact same match.<br>
But if the predicted cap is “A little dog is running on green grass”, the BLEU score will be around 0.75. Model shown average BLEU score of 0.58 over 100 images.<br>

## Outcome <br>
Following are few results of my model predicting captions for images

<img src="https://github.com/raivarsha14/ImageCap/blob/master/images/Screen%20Shot%202020-07-02%20at%207.12.14%20PM.png"
     alt="Good Results"
     style="float: left; margin-right: 10px;" />

Following are few results of my model predicting captions where it get confused with the object and colours

<img src="https://github.com/raivarsha14/ImageCap/blob/master/images/Screen%20Shot%202020-07-02%20at%207.12.42%20PM.png"
     alt="Confused Results"
     style="float: left; margin-right: 10px;" />

Above we can see, model wrongly predicted red scooter, whereas the actual colour of scooter is yellow. A "bus" is wrongly predicted as "double decker bus"

## Conclusion & Future Directions <br>

While working on this project, I found this trend with my model that it was predicting a “bus” as “red double-decker bus” for any bus picture or for “giraffe” model was predicted “two giraffe” or with the “plate of food” model predicted that as “plate of food with a cup of tea/coffee”. Those I guess is because of the training images I feed-in for training must have a very number of “red double-decker bus” where there are buses in the image, majority the images of “giraffe” might be “two giraffes” etc. To overcome with this issue we can try batch normalization.<br>
To get extremely high accuracy, it is very important to train the model on a very large variety of objects, actions, colours, etc. I used “ImageNet” weight for feature extraction of images, where there is only 1000 class classification of images. I used more than 118k images to train the model, that means if I use more than 1000 class classification and then train the model that can predict more efficiently.<br>
Development of mobile application using the final model can be useful for real time implementation of caption generation.
