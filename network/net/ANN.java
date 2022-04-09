package neuralnet.network.net;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.network.layer.FullyConnected;
import neuralnet.network.layer.Layer;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN extends Network {
	int[] structure = NetworkOrganizer.structure;
	int length = structure.length;
	
	/**
	 * Creates an artificial neural network
	 * */
	public ANN() {
		layers = new Layer[length];
		for(int i = 1; i < length - 1; i++){
			layers[i] = new FullyConnected(structure[i-1], structure[i], FullyConnected.RELU_ACTIVATION);
		}
		layers[length - 1] = new FullyConnected(structure[length - 2], structure[length - 1], FullyConnected.SOFTMAX_ACTIVATION);
	}
	
}
