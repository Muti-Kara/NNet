package neuralnet.network.net;

import neuralnet.network.layer.FullyConnected;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN extends Net {

	public static final int RELU = 0;
	public static final int SOFTMAX = 1;
	
	/**
	 * Creates an artificial neural network
	 * */
	public ANN() {
		layers.add(new FullyConnected(0, 0, 0));
	}

	@Override
	public void addLayer(int type, int... layerDescriptor) {
		layers.add(new FullyConnected(type, layerDescriptor[0], layerDescriptor[1], type));
	}
	
	@Override
	public String toString() {
		String str = (layers.size() - 1) + "\n";
		for(int i = 1; i < layers.size(); i++){
			str += "\n" + layers.get(i).toString();
		}
		return str;
	}

}
