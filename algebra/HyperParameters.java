package algebra;

/**
* HyperParameters
* A class for global variables and hyper parameters of network.
* @author Muti Kara
*/
public class HyperParameters {
	/**
	 * Project directory.
	 * */
	public static String projectDir = "/home/yuio/project/neuralnet";
	
	/**
	 * Structure of neural network.
	 * structure[] is for ann.
	 * convolutional, kernel, pool is for cnn.
	 * */
	public static int[] structure = new int[]{256, 26};
	public static int[] convolutional = new int[]{2, 2};
	public static int[] kernel = new int[]{5, 3};
	public static int[] pool = new int[]{2, 2};
	
	/**
	 * Size of our training data.
	 * Size of our test data.
	 * */
	public static int DATA_SIZE;
	public static int TEST_SIZE;
	
	/**
	 * Number of epochs to train network.
	 * */
	public static int EPOCH = 300;
	public static int STOCHASTIC = 200;
	public static int CNN_EPOCH = 5;
	public static int CNN_GENERATION = 20;
	
	/**
	 * Image size before putting into network.
	 * */
	public static int IMAGE_SIZE = 37;
	
	/**
	 * Learning rate and momentum factor of artificial neural network
	 * Learning rate of CNN
	 * */
	public static double LEARNING_RATE = 0.03;
	public static double CNN_LEARNING_RATE = 0.05;
	public static double MOMENTUM_FACTOR = 0.01;
	
	/**
	 * Random values for initialiation of ANN and CNN.
	 * */
	public static double RANDOMIZATION = 0.001;
	public static double KERNEL_RANDOMIZATION = 0.1;
}
