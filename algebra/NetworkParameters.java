package algebra;

/**
* NetworkParameters
* A class for global variables and hyper parameters of network.
* @author Muti Kara
*/
public class NetworkParameters {
	/**
	 * Project directory.
	 * */
	public static String readDir = "/home/yuio/project/neuralnet/preproccess/";
	
	/**
	 * Structure of neural network.
	 * structure[] is for ann.
	 * convolutional, kernel, pool is for cnn.
	 * */
	public static int[] structure = new int[0];
	public static int[] convolutional = new int[0];
	public static int[] kernel = new int[0];
	public static int[] pool = new int[0];
	
	public static void addFullyConnectedLayer(int numOfNeurons) {
		int[] newStructure = new int[structure.length + 1];
		newStructure[structure.length] = numOfNeurons;
		System.arraycopy(structure, 0, newStructure, 0, structure.length);
		structure = newStructure;
	}
	
	public static void addConvolutionalLayer(int numOfKernels, int kernelSize, int poolingSize) {
		int[] newConvolutional = new int[convolutional.length + 1];
		int[] newKernel = new int[kernel.length + 1];
		int[] newPool = new int[pool.length + 1];
		
		newConvolutional[convolutional.length] = numOfKernels;
		newKernel[kernel.length] = kernelSize;
		newPool[pool.length] = poolingSize;
		
		System.arraycopy(convolutional, 0, newConvolutional, 0, convolutional.length);
		System.arraycopy(kernel, 0, newKernel, 0, kernel.length);
		System.arraycopy(pool, 0, newPool, 0, pool.length);
		
		convolutional = newConvolutional;
		kernel = newKernel;
		pool = newPool;
	}
	/**
	 * Size of our training data.
	 * Size of our test data.
	 * */
	public static int dataSize;
	public static int testSize;
	
	/**
	 * Number of epochs to train network.
	 * */
	public static int epoch = 300;
	public static int stochastic = 200;
	
	public static void setEpoch(int epoch) {
		NetworkParameters.epoch = epoch;
	}
	
	public static void setStochastic(int stochastic) {
		NetworkParameters.stochastic = stochastic;
	}
	
	public static int cnnEpoch = 5;
	public static int cnnGeneration = 40;
	
	public static void setCnnEpoch(int cnnEpoch) {
		NetworkParameters.cnnEpoch = cnnEpoch;
	}
	
	public static void setCnnGeneration(int cnnGeneration) {
		NetworkParameters.cnnGeneration = cnnGeneration;
	}
	
	/**
	 * Image size before putting into network.
	 * */
	public static int imageSize = 33;
	
	public static void setImageSize(int imageSize) {
		NetworkParameters.imageSize = imageSize;
	}
	
	/**
	 * Learning rate and momentum factor of artificial neural network
	 * Learning rate of CNN
	 * */
	public static double learningRate = 0.03;
	public static double cnnLearningRate = 0.05;
	public static double momentumFactor = 0.01;
	
	public static void setLearningRate(double learningRate) {
		NetworkParameters.learningRate = learningRate;
	}
	
	public static void setCnnLearningRate(double cnnLearningRate) {
		NetworkParameters.cnnLearningRate = cnnLearningRate;
	}
	
	public static void setMomentumFactor(double momentumFactor) {
		NetworkParameters.momentumFactor = momentumFactor;
	}
	
	/**
	 * Random values for initialiation of ANN and CNN.
	 * */
	public static double randomization = 0.001;
	public static double kernelRandomization = 0.1;
	
	public static void setRandomization(double randomization) {
		NetworkParameters.randomization = randomization;
	}
	
	public static void setKernelRandomization(double kernelRandomization) {
		NetworkParameters.kernelRandomization = kernelRandomization;
	}
}
