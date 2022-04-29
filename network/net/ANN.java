package neuralnet.network.net;

import neuralnet.network.layer.FullyConnected;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN extends Net {
	
	/**
	 * Creates an artificial neural network
	 * */
	public ANN() {
		layers.add(new FullyConnected(0, 0, -1));
	}

	@Override
	public void addLayer(int type, int... layerDescriptor) {
		layers.add(new FullyConnected(layerDescriptor));
	}

}
