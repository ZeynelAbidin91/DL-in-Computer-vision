from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds

def next_batch(X, y, batchSize):
    # loop over our dataset 'X' in mini-batches, yielding a tuple of
    # the current batched data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=float, default=100, help='# of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01, help='learning rate')
ap.add_argument('-b', '--batch_size', type=int, default=32, help='Size of SGD mini-batch')
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1,000 data points, where each data point
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=2)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature matrix -- this little trick allows
# us to treat the bias as a trainable parameter within the weight matrix
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):
    # initialize the total loss for the epoch
    epochLoss = []

    # loop over our data in batches
    for (batchX, batchY) in next_batch(X, y, args['batch_size']):
        # take the dot product between our currrent batch of features
        # and the weight matrix, then pass this value throough our
        # activate function
        preds = sigmoid_activation(batchX.dot(W))

        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))


        gradient = batchX.T.dot(error)
        W += -args['alpha'] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch == 0 or (epoch+1) % 5 == 0:
        print('[INFO] epoch={}, loss={:.7f}'.format(int(epoch + 1), loss))
    
print('[INFO] evaluating...')
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.title('Training Loss')
plt.plot(np.arange(0, args['epochs']), losses)
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()