This is a neural network library for my cs102 project. It will be used for image recognition (character recognition). <br />
This network consists of two part: <br />
1- Convolutional Layers. These layers will be trained with genetic algorithm. <br />
2- Fully Connected Layers. These layers will be trained with backpropagation algorithm. <br />
My intended network will have a structure:<br />
1 Convolutional Layer with 2 5x5 kernels<br />
1 Max Pool Layer<br />
1 Convolutional Layer with 2 3x3 kernels<br />
1 Max Pool Layer<br />
1 Fully Connected Layer with neuron size of 256 (INPUT)<br />
1 Fully Connected Layer with neuron size of 128 (RELU)<br />
1 Fully Connected Layer with neuron size of 26 (SOFTMAX)<br />
