<h1>Neural Network Implementation (NeuralNetwork.py)</h1>
This Python module (NeuralNetwork.py) provides a basic implementation of a neural network. It includes various activation functions, dropout functionality, and layer management. Below, we’ll explore the key components of this neural network.

<h2>Components</h2>
<ol>
1. Activation Functions:<ul>
<li>The module defines several activation functions:</li><ul>
<li><strong>ReLU (Rectified Linear Unit):</strong> Used for hidden layers.</li>
<li><strong>Softmax:</strong> Typically used for the output layer in multiclass classification problems. (in my implementation will work only as such)</li>
<li><strong>Sigmoid:</strong> Commonly used for binary classification.</li>
<li><strong>Tanh (Hyperbolic Tangent):</strong> Another option for hidden layers.</li></ul></ul>
2. Layer Class:<ul>
<li>The Layer class represents a single layer in the neural network.</li>
<li>It includes methods for forward and backward propagation.</li>
<li><strong>Dropout</strong> functionality can be enabled or disabled for each layer.</li></ul>
3. NeuralNetwork Class:<ul>
<li>The NeuralNetwork class manages the entire network.</li>
<li>You can add layers, set dropout probabilities, and train the network.</li>
<li>It supports both random and fixed dropout during training.</li>
</ol>

<h2>Usage</h2>
<h3>1. Creating a Neural Network:</h3>
```
from NeuralNetwork import NeuralNetwork, Relu, Softmax

nn = NeuralNetwork(alpha=0.01, input_size=10, dropout_probability=0.3)
nn.add_layer(n=64, activation_function=Relu())
nn.add_layer(n=32, activation_function=Relu())
nn.add_layer(n=3, activation_function=Softmax())
```
<h3>2. Training the Network:</h3>
```
#Add training data (input-output pairs)
nn.add_series(input_data, output_data)

#Train for a specified number of epochs
nn.do_epochs(epochs=10)
```
<h3>3. Making Predictions:</h3>
```

input_sample = [0.1, 0.2, ..., 0.9]
predicted_output = nn.predict(input_sample)
```