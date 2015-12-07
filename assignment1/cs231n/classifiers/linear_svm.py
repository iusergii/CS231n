import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0

  for i in xrange(num_train):
    scores = W.dot(X[:, i])  #[ 0.5554988   0.15474304 -0.67970684  0.41462399 -0.90796891  0.93243053  1.72867754  0.19333425  0.5069284   0.11827186]
    correct_class_score = scores[y[i]] # 1.72867754115
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      
      if margin > 0:
        loss += margin
        # 
        dW[j, :] += X[:, i].T
        dW[y[i], :] -= X[:, i].T


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(W, X) # 10 x 49000

  # score_ones = np.ones(scores.shape)   # 10 X 49000 [[1...1]...[1...1]]
  # create matrix with colum = y[i]:  array([[ 2.,  6., ... 7.],
                                          #  [ 2.,  6.,  ...7.],
                                          #  [ 2.,  6.,  ...7.]])
  y_score = np.ones(scores.shape) * scores[y, np.arange(0, scores.shape[1])] # 10 X 49000
  
  deltas = np.ones(scores.shape)      # set delta = 1
  margin = scores - y_score + deltas  

  margin[ margin < 0 ] = 0
  margin[y, np.arange(0, scores.shape[1])] = 0 # exclude loss for correct class

  loss = np.sum(margin)
  
  # Average over number of training examples
  num_train = X.shape[1]
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[ margin > 0 ] = 1 # mark features which create loss
  margin[y, np.arange(0, scores.shape[1])] = -1 * np.sum(margin, axis=0) # gradient for w_yi
  dW = np.dot(margin, X.T)

  # Average over number of training examples
  dW /= num_train
  # Add regularization to gradient
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
