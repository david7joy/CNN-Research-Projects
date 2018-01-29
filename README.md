# CNN-Research-Projects
A repository for my CNN Research &amp; Projects

This project is essentially my 0 to something on CNN !!

Train Image Category Classifier 

Convolution Neural Network !! 

2012 - > Alex Krizhevsky used CNN 

Kaggle !!
https://www.kaggle.com/c/dogs-vs-cats/data

https://github.com/llSourcell/how_to_make_an_image_classifier/blob/master/demo.ipynb


https://www.mathworks.com/help/vision/ref/trainimagecategoryclassifier.html

https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html


CONVOLUTION NEURAL NETWORKS !! 

https://www.youtube.com/watch?v=FTr3n7uBIuE

http://neuralnetworksanddeeplearning.com/chap6.html

Great CNN Blog for reading.

http://timdettmers.com/2015/03/26/convolution-deep-learning/


For ANN modeling (or other ML algorithms)  we typically partition the dataset into three parts: a training set (say, 60%), a validation set (e.g. 20%) and a test set (e.g. 20%). Normally, you are training the network with the training set to adjust the weights. To make sure you don't overfit the network and also fine-tune models you need to input the validation set to the network and check if the error is within some range (This set is not being using directly to adjust the weights but used to give the optimal number of hidden units or determine a stopping point for the back-propagation algorithm). Finally, the accuracy of the model on the test data gives a realistic estimate of the performance of the model on completely unseen data and in order to confirm the actual predictive power of the network.
To summarize all:
Training set  --> to fit the parameters [i.e., weights]
Validation set --> to tune the parameters [i.e., architecture]
Test set --> to assess the performance [i.e., generalization and predictive power]

Batch size defines number of samples that going to be propagated through the network.
For instance, let's say you have 1050 training samples and you want to set up batch_size equal to 100. Algorithm takes first 100 samples (from 1st to 100th) from the training dataset and trains network. Next it takes second 100 samples (from 101st to 200th) and train network again. We can keep doing this procedure until we will propagate through the networks all samples. The problem usually happens with the last set of samples. In our example we've used 1050 which is not divisible by 100 without remainder. The simplest solution is just to get final 50 samples and train the network.
Advantages:
•	It requires less memory. Since you train network using less number of samples the overall training procedure requires less memory. It's especially important in case if you are not able to fit dataset in memory.
•	Typically networks trains faster with mini-batches. That's because we update weights after each propagation. In our example we've propagated 11 batches (10 of them had 100 samples and 1 had 50 samples) and after each of them we've updated network's parameters. If we used all samples during propagation we would make only 1 update for the network's parameter.
Disadvantages:
•	The smaller the batch the less accurate estimate of the gradient. In the figure below you can see that mini-batch (green color) gradient's direction fluctuates compare to the full batch (blue color).
 
Stochastic is just a mini-batch with batch_size equal to 1. Gradient changes its direction even more often than a mini-batch.

Why padding ? 
1) to avoid size of image from shriking 
2) to use corner pixels and information more usefully 

Padding ->  n + 2p - f +1, where n is size of image 144X144 then n = 144, p is padding size , 
            f is size of kernal i.e 3X3, f=3
Stride - > z =[(( n + 2p - f )/ s) + 1] , where  s is stride 
           sometimes the overall result may not be an integer hence we need to floor the whole value !! ie floor(z) 
