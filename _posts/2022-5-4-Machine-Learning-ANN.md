```markdown
layout:     post
title:      "Machine Learning and Predictive Analytics: Introduction to Artificial Neural Networks with Keras"
subtitle:   "Introduction to dimensionality reduction and python implementation"
date:       2022-5-4 18:00:00
author:     "Haohan"
catalog: true
header-style: text
mathjax: true
tags:

  - machine learning
  - lecture notes
  - model
```

# Importance

## **Artificial Neural Networks (ANN) **

An artificial neuron is a mathematical function based on a model of biological neurons, where each neuron takes inputs, weighs them separately, sums them up and passes this sum through a nonlinear function to produce output. 

- A neuron is a mathematical function modeled on the working of biological neurons
- It is an elementary unit in an artificial neural network
- One or more inputs are separately weighted
- Inputs are summed and passed through a nonlinear function to produce output
- Every neuron holds an internal state called activation signal
- Each connection link carries information about the input signal
- Every neuron is connected to another neuron via connection link

## **Percepton**

A Perceptron is an algorithm for supervised learning of binary classifiers.

There are two types of Perceptrons: Single layer and Multilayer.

- Single layer - Single layer perceptrons can learn only linearly separable patterns
- [Multilayer](https://www.simplilearn.com/tutorials/deep-learning-tutorial/multilayer-perceptron) - Multilayer perceptrons or feedforward neural networks with two or more layers have the greater processing power

- based on a **threshold logic unit** (TLU), also called a linear threshold unit LTU

- TLU computes a weighted sum of its inputs, then applies a step function

  ![](/img/in-post/perceptron.png) 

  [Percepton](https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron#:~:text=A%20Perceptron%20is%20a%20neural%20network%20unit%20that%20does%20certain,value%20%E2%80%9Df(x))

$$
h_{w,b}(X)  = \phi(XW+b)
$$

* **X** - matrix of input features, one row per instance, one column per feature
*  **W** - contains all the connection weights except for the one from the bias neuron, one row per input neuron, one column per unit in the layer 
* **b** contains all the connection weights between the bias neuron and the neurons in the output layer. It has one bias term per neuron in the output layer.
* $\phi$ - the activation function: when the artificial neurons are TLUs, it is a step function 

- **The Perceptron learning rule **

  - Perceptrons do not output a class probability; rather, they make predictions based on a hard threshold. 
  - Perceptron, like any linear classifier, cannot find the correct solution if the classes are not linearly separable. 

- **Multilayer Perceptron and Backpropagation **

  - An MLP is composed of one (passthrough) **input layer**, one or more layers of TLUs, called **hidden layers**, and one final layer of TLUs called the **output layer**

    ![](/img/in-post/MLP.png) 

  - Every layer except the output layer includes a bias neuron and is fully connected to the next layer. 

  - The signal flows only in one direction, **feedforward neural network (FNN)**. 

## **Backpropagation** 

- Backpropagation is an algorithm used in machine learning that works by calculating the **gradient of the loss function**, which points us in the direction of the value that minimizes the loss function. It relies on the **chain rule** of calculus to calculate the gradient backward through the layers of a neural network. Using gradient descent, we can iteratively move closer to the minimum value by taking small steps in the direction given by the gradient.

  [backpropagation 1](https://programmathically.com/understanding-backpropagation-with-gradient-descent/)

  [backpropagation 2](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)

  Activation functions 

- calculates a “weighted sum” of its input, adds a bias and then decides whether it should be “fired” or not

- Activation function A = “activated” if Y > threshold else not

  [activation function](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

  ![](/img/in-post/activation.png) 

  ![](/img/in-post/activation2.png) 

## **Regression MLPs** 

- output always positive: ReLU activation 

- Alternative: $softplus(z) = log(1 + exp(z)) $, It is close to 0 when z is negative, and close to z when z is positive. 

- fall in given range

  - 0 to 1: logistic function 
  - -1 to 1: hyperbolic tangent 

- Loss Function:

  - typical: **MSE**
  - lots of outliers: **mean absolute error**
  - **Huber loss**: combination of both

  - The Huber loss is **quadratic** when the error is **smaller than a threshold δ** (typically 1) but **linear** when the **error is larger than δ.** 

- **Implementation**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
```



```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))

myutils.tic()
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
myutils.toc()

mse_test = model.evaluate(X_test, y_test)
print("mse_test: ", mse_test)

X_new = X_test[:3]
y_pred = model.predict(X_new)
```



## **Classification MLPs** 

- MLPs can also easily handle multilabel binary classification tasks 
- Note that the output probabilities do not necessarily add up to 1 
- If each instance can belong only to a single class, out of three or more possible classes :
  - use the **softmax** activation function 
  - The softmax function will ensure that all the estimated probabilities are between 0 and 1 and that they add up to 1 (which is required if the classes are **exclusive**). 
- **cross-entropy loss** (also called the log loss )
- **Implementation**

```python
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

model.evaluate(X_test, y_test)
```



## **Saving and Restoring a Model, Callback**

- Keras will use the HDF5 format to save both the model’s architecture (including every layer’s hyperparameters) and the values of all the model parameters for every layer (e.g., connection weights and biases). It also saves the optimizer (including its hyperparameters and any state it may have) 
- The fit() method accepts a callbacks argument that lets you specify a list of objects that Keras will call at the start and end of training, at the start and end of each epoch, and even before and after processing each batch. For example, the **ModelCheckpoint** callback saves checkpoints of your model at regular intervals during training, by default at the end of each epoch: 
- If you use a validation set during training, you can set **save best only=True** when creating the ModelCheckpoint. In this case, it will only save your model when its performance on the validation set is the best so far 
- Another way to implement early stopping is to simply use the **EarlyStopping** callback. It will interrupt training when it measures no progress on the validation set for a number of epochs (defined by the patience argument), and it will optionally roll back to the best model. 
- There is no need to restore the best model saved because the EarlyStopping callback will keep track of the best weights and restore them for you at the end of training 

```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)

model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")

```

```python
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

```

## **TensorBoard** 

great **interactive visualization** tool that you can use to view the **learning curves** during training, compare learning curves between multiple runs, visualize the computation graph, analyze training statistics, view images generated by your model, visualize complex multidimensional data projected down to 3D and automatically clustered 

```python
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
myutils.tic()
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])
```



## **Fine-Tuning Neural Network Hyperparameters** 

-  it’s often simpler and more efficient to pick a model with **more layers and neurons than you actually need**, then use **early stopping** and other regularization techniques to prevent it from overfitting. 
- In general you will get more bang for your buck by increasing the number of layers instead of the number of neurons per layer 
- Learning rate - the optimal learning rate is about half of the maximum learning rate - the learning rate above which the training algorithm diverges 

- Optimizer - there are other optimizers besides Mini-batch Gradient Descent and we shall consider them next time 

- Batch size - can have a significant impact on your model’s performance and training time. The main benefit of using large batch sizes is that hardware accelerators like GPUs can process them efficiently, so the training algorithm will see more instances per second. On the other hand, for large batch sizes your data might not fit into GPU memory. Also large batch sizes often lead to training instabilities, especially at the beginning of training, and the resulting model may not generalize as well as a model trained with a small batch size. 
- Activation function - in general, the ReLU activation function will be a good default for all hidden layers. For the output layer, it really depends on your task. 

- Number of iterations - in most cases does not actually need to be tweaked: just use early stopping instead. 

```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100)               .tolist(),
    "learning_rate": reciprocal(3e-4, 3e-2)      .rvs(1000).tolist(),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
myutils.tic()
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
myutils.toc()

rnd_search_cv.score(X_test, y_test)

model = rnd_search_cv.best_estimator_.model
model

model.evaluate(X_test, y_test)
```

