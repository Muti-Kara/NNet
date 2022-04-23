package neuralnet.network.net;

import neuralnet.network.layer.FullyConnected;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN extends Net{
	
	/**
	 * Creates an artificial neural network
	 * */
	public ANN() {
		layers.add(new FullyConnected(0, 0, -1));
		for(int i = 1; i < structure.size(); i++){
			int activation = (i == structure.size() - 1)? FullyConnected.SOFTMAX : FullyConnected.RELU;
			layers.add(new FullyConnected(structure.get(i-1), structure.get(i), activation));
		}
	}
	
}
