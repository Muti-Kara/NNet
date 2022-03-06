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
	public final static String projectDir = "/home/yuio/project/neuralnet";
	
	/**
	 * Structure of neural network.
	 * */
	public final static int[] structure = new int[]{288, 125, 3};
	public final static int[] convolutional = new int[]{2};
	public final static int[] kernel = new int[]{3};
	public final static int[] pool = new int[]{2};
	
	/**
	 * Size of our training data.
	 * Size of our test data.
	 * */
	public final static int DATA_SIZE = 6;
	public final static int TEST_SIZE = 3;
	
	/**
	 * Number of epochs to train network.
	 * */
	public final static int EPOCH = 1000;
	
	/**
	 * Image size before putting into network.
	 * */
	public final static int IMAGE_SIZE = 25;
	
	/**
	 * Learning rate of network and randomization coefficient for
	 * initial values of weights and biases.
	 * */
	public final static double LEARNING_RATE = 0.000001;
	public final static double RANDOMIZATION = 0.01;
	public final static double KERNEL_RANDOMIZATION = 1;
}
