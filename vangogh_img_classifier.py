import numpy as np
import pandas as pd
import tensorflow as tf

# read data and split into training and development datasets

df = pd.read_csv('/Users/arjunsatheesan/Downloads/vangogh/img_classification_train.csv')
test_df = pd.read_csv('/Users/arjunsatheesan/Downloads/vangogh/img_classification_test.csv')
train_df = df.sample(frac=0.8)
dev_df = df.drop(train_df.index)


# split the train and dev datasets into their corresponding features and labels

train_labels = train_df.as_matrix(columns=['label'])
train_features = train_df.drop(labels='label', axis=1).as_matrix()

dev_labels = dev_df.as_matrix(columns=['label'])
dev_features = dev_df.drop(labels='label', axis=1).as_matrix()

test_labels = test_df.as_matrix(columns=['label'])
test_features = test_df.drop(labels='label', axis=1).as_matrix()



# parameters

epochs= 10000
batch_size=240
display_step = 100
feature_size = 2352 #28 x 28 x 3
class_size=2
epoch_index=0
epochs_completed=0



# change the format of the labels according to one-hot encoding

def one_hot_encode(labels, classes=2):
    one_hot = np.zeros([len(labels), classes])
    for i in range(len(labels)):
        one_hot[i, labels[i]] = 1.
    return one_hot

train_labels = one_hot_encode(train_labels)
dev_labels = one_hot_encode(dev_labels)
test_labels = one_hot_encode(test_labels)


# Computation graph construction

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 3], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,3], strides=[1,2,2,3], padding='SAME')
 

def compute_accuracy(features, labels, keep_prob):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={x: features,keep_probability:keep_prob})
    correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: features, y: labels,keep_probability: keep_prob})
    return result

def next_batch(batch_size):
    
    global train_features
    global train_labels
    global epoch_index
    global epochs_completed
    
    start = epoch_index
    epoch_index += batch_size
       
    if epoch_index > len(train_features):
        epochs_completed += 1
        p = np.random.permutation(len(train_features))
        train_features = train_features[p]
        train_labels = train_labels[p]
# start next epoch
        start = 0
        epoch_index = batch_size
        assert batch_size <= len(train_features)
    end = epoch_index
    return train_features[start:end], train_labels[start:end]


# create the convnet model

x = tf.placeholder(tf.float32,[None, feature_size])
y = tf.placeholder(tf.float32,[None, class_size])
keep_probability = tf.placeholder(tf.float32)
x_reshape = tf.reshape(x, shape=[-1, 28, 28, 3])

# First convolution layer. Weight variable has patch size of 5x5, in-size of 1, and out-size of 32
# Each layer includes convolution followed by maxpooling and dropout


W_conv1 = weight_variable([3,3,3,32]) 
b_conv1 = bias_variable([32])
conv1 = conv2d(x_reshape, W_conv1, b_conv1)
conv1 = maxpool2d(conv1) 
conv1 = tf.nn.dropout(conv1, keep_probability)

# Second convolution layer

W_conv2 = weight_variable([3,3,32,64]) 
b_conv2 = bias_variable([64])
conv2 = conv2d(conv1, W_conv2, b_conv2)
conv2 = maxpool2d(conv2)
conv2 = tf.nn.dropout(conv2, keep_probability)

# Third convolution layer

W_conv3 = weight_variable([3,3,64,128]) 
b_conv3 = bias_variable([128])
conv3 = conv2d(conv1, W_conv3, b_conv3)
conv3 = maxpool2d(conv3)
conv3 = tf.nn.dropout(conv3, keep_probability)

# Fully connected layer 1
# Reshape conv3 output to fit fully connected layer input

W_fc1 = weight_variable([28 * 28 * 128, 128])
b_fc1 = bias_variable([128])
fc1 = tf.reshape(conv3, [-1, 28 * 28 * 128])
fc1 = tf.add(tf.matmul(fc1, W_fc1), b_fc1)
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_probability)


# Fully connected layer 2
# Reshape FC1 output to fit fully connected layer input

W_fc2 = weight_variable([28 * 28 * 128, 256])
b_fc2 = bias_variable([256])
fc2 = tf.reshape(conv3, [-1, 28 * 28 * 128])
fc2 = tf.add(tf.matmul(fc2, W_fc2), b_fc2)
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, keep_probability)

# Output layer

W_fc3 = weight_variable([256,class_size])
b_fc3 = bias_variable([class_size])
prediction = tf.nn.softmax(tf.matmul(fc2, W_fc3) + b_fc3)


# Loss function and optimizer

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

# Initialize all the variables
init = tf.global_variables_initializer()

# Run session for the computation graph

sess = tf.Session()
sess.run(init)


# batch_features, batch_labels = train_features[:batch_size], train_labels[:batch_size]

for i in range(epochs):
    batch_features, batch_labels = next_batch(batch_size)
    
#     sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_probability: 0.75})
    sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_probability: 0.75})
    if i%display_step ==0:
        print("Training accuracy for epoch#: ",compute_accuracy(batch_features,batch_labels,0.75))
       

print("Development accuracy: ",compute_accuracy(dev_features,dev_labels,1.0))
print("Test accuracy: ",compute_accuracy(test_features,test_labels,1.0))


