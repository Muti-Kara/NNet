package algebra;

/**
* HyperParameters
* A class for global variables and hyper parameters of network.
* @author Muti Kara
*/
public class HyperParameters {
	/**
	 * Project directory is here.
	 * */
	public static String projectDir = "/home/yuio/project/neuralnet";
	
	/**
	 * Structure of neural network.
	 * */
	public static int[] structure = new int[]{196, 75, 5};
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
	public static int EPOCH = 15000;
	
	/**
	 * Image size before putting into network.
	 * */
	public static int IMAGE_SIZE = 33;
	
	/**
	 * Learning rate of network and randomization coefficient for
	 * initial values of weights and biases.
	 * */
	public static double LEARNING_RATE = 0.003;
	public static double RANDOMIZATION = 0.002;
	public static double KERNEL_RANDOMIZATION = 0.4;
}
