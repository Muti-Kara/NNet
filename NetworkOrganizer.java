package neuralnet;

/**
* NetworkParameters
* A class for global variables and hyper parameters of network.
* @author Muti Kara
*/
public class NetworkOrganizer {
	
	/**
	 * Number of epochs to train network.
	 * */
	public static int epoch;
	public static int stochastic;
	
	public static void setEpoch(int epoch) {
		NetworkOrganizer.epoch = epoch;
	}
	
	public static void setStochastic(int stochastic) {
		NetworkOrganizer.stochastic = stochastic;
	}
	
	public static int cnnEpoch;
	public static int cnnGeneration;
	
	public static void setCnnEpoch(int cnnEpoch) {
		NetworkOrganizer.cnnEpoch = cnnEpoch;
	}
	
	public static void setCnnGeneration(int cnnGeneration) {
		NetworkOrganizer.cnnGeneration = cnnGeneration;
	}
	
	/**
	 * Learning rate and momentum factor of artificial neural network
	 * Learning rate of CNN
	 * */
	public static double annLearningRate;
	public static double cnnLearningRate;
	public static double momentumFactor;
	
	public static void setLearningRate(double learningRate) {
		NetworkOrganizer.annLearningRate = learningRate;
	}
	
	public static void setCnnLearningRate(double cnnLearningRate) {
		NetworkOrganizer.cnnLearningRate = cnnLearningRate;
	}
	
	public static void setMomentumFactor(double momentumFactor) {
		NetworkOrganizer.momentumFactor = momentumFactor;
	}
	
	/**
	 * Random values for initialiation of ANN and CNN.
	 * */
	public static double randomization;
	
	public static void setRandomization(double randomization) {
		NetworkOrganizer.randomization = randomization;
	}
}
