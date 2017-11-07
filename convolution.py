
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf


x = np.linspace(-3.0,3.0,100) # Linear space using numpy 
x # This will return a array 
x.shape # provides shape 
x.dtype # data type 

x = tf.linspace(-3.0,3.0,100) # linear space using tensor flow
y = tf.add(675,325)
print(x)
print(y)
# this will return the a tf tensor with name linspace_3:0 whenever we :0
# that means an output of linspace_3, another difference it that it only 
# outputs a tensor, it doesn't out put values as it has not yet calculated the values
# this is a basic tf operation which has been added to tf default computational graph, the result of that 
# operation is the tensor that we have returned 

g = tf.get_default_graph()
g = tf.reset_default_graph()
[op.name for op in g.get_operations()] # show all the operations that have been added to the default graph
g.get_tensor_by_name('LinSpace_3'+':0') # the result of a tf operation is tf tensor 
g.get_tensor_by_name('Add'+':0')

# to compute anything in tension flow we need a tf session

sess = tf.Session() # by default takes default graph 
computed_x = sess.run(x)
computed_x

# or we can do the below

computed_x = x.eval(session=sess)
computed_x

computed_y = y.eval(session=sess)
computed_y
sess.close()

#create a interactive session 
sess = tf.InteractiveSession()
x.eval()   # this way we can speeden up the process 

x.get_shape().as_list() # as a list !!

# making a gaussian kernal - also referred to as normal curve or bell curve - Mean is 0 
mean = 0
sigma = 1.0

z = (tf.exp(tf.negative(tf.pow(x-mean,2.0)/(2.0*tf.pow(sigma,2.0))))*(1.0/(sigma*tf.sqrt(2.0*3.1415)))) 

res = z.eval()
plt.plot(res)

# make a 2D gausian - this can be done by multiplying the vector by its transpose 

ksize = z.get_shape().as_list()[0]
z_2d = tf.matmul(tf.reshape(z,[ksize,1]),tf.reshape(z,[1,ksize]))

plt.plot(z_2d.eval()) 
plt.imshow(z_2d.eval()) 

# reading an image using matplot lib
img1=mpimg.imread('cat.1.jpg')
plt.imshow(img1,cmap='gray')
print(img1.shape);

img = data.camera().astype(np.float32)
plt.imshow(img,cmap='gray')
print(img.shape)

# using reshape function 
img_4d = img.reshape([1,img.shape[0],img.shape[1],1])
print(img_4d.shape)

#  we can also use tensorflow reshape function 
img_4d = tf.reshape(img,[1,img.shape[0],img.shape[1],1])
print(img_4d.shape)

# use other function on the 
img_4d.get_shape()
img_4d.get_shape().as_list()

# Convolution Kernal 
z_4d = tf.reshape(z_2d,[ksize,ksize,1,1]) # h and w of kernal with channel and no of filters
z_4d.get_shape().as_list()

convolve = tf.nn.conv2d(img_4d,z_4d,strides=[1,1,1,1],padding='SAME') 
#Stride decides how to move and padding decides what to do with the border 
res = convolve.eval()

print(res.shape)

#Removes single-dimensional entries from the shape of an array.
plt.imshow(np.squeeze(res),cmap='gray') 

#or 

plt.imshow(res[0,:,:,0])

#Gabor Kernal - sinewave + Gausian combination 

xs = tf.linspace(-3.0,3.0,ksize)  

ys = tf.sin(xs)

plt.figure() 

plt.plot(ys.eval())

ys = tf.reshape(ys, [ksize,1])

ones = tf.ones((1,ksize))

wave = tf.matmul(ys,ones)

plt.imshow(wave.eval(),cmap='gray')

gabor = tf.matmul(wave,z_2d)

plt.imshow(gabor.eval())

convolve_g = tf.nn.conv2d(img_4d,gabor,strides=[1,1,1,1],padding='SAME')

# Placeholder 

img= tf.placeholder(tf.float32,shape=[None,None],name='img')

img_3d = tf.expand_dims(img,2)

dims = img_3d.get_shape()

img_4d = tf.expand_dims(img_3d,0)

print(img_4d.get_shape().as_list())

mean = tf.placeholder(tf.float32,name='mean')
sigma = tf.placeholder(tf.float32,name='sigma')
ksize = tf.placeholder(tf.float32,name='ksize')


x = tf.linspace(-3.0,3.0,ksize)  
z = (tf.exp(tf.negative(tf.pow(x-mean,2.0)/
                        (2.0*tf.pow(sigma,2.0))))*(1.0/(sigma*tf.sqrt(2.0*3.1415))))
z_2d = tf.matmul(tf.reshape(z,[ksize,1]),tf.reshape(z,[1,ksize]))
ys = tf.sin(xs)
ys = tf.reshape(ys, tf.pack([ksize,1]))
ones = tf.ones(tf.pack([1,ksize]))
wave = tf.matmul(ys,ones)
gabor = tf.matmul(wave,z_2d)
gabor_4d = tf.reshape(gabor,tf.pack([ksize,ksize,1,1]))
convolved_g = tf.nn.conv2d(img_4d,gabor_4d,strides=[1,1,1,1],padding='SAME',name='convolved')
convolved_img = convolved_g[0,:,:,0]

res = convolved_img.eval(feed_dict={img:data.camera(),mean:0.0,sigma:1.0,ksize:100})



