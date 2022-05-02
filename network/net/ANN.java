package nnet.network.net;

import nnet.network.layer.FullyConnected;

/**
 * A Fully Connected network
 * @author Muti Kara
 * */
public class ANN extends Net {
	public static final int INPUT = -1;
	public static final int RELU = 0;
	public static final int SOFTMAX = 1;
	
	int previous = 0;
	
	/**
	 * Overrides addLayer method.
	 * */
	@Override
	public void addLayer(int type, int... layerDescriptor) {
		layers.add(new FullyConnected(type, previous, layerDescriptor[0], type));
		previous = layerDescriptor[0];
	}
	
}
