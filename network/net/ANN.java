package neuralnet.network.net;

import neuralnet.network.layer.FullyConnected;

/**
 * A FC network
 * @author Muti Kara
 * */
public class ANN extends Net {
	public static final int INPUT = -1;
	public static final int RELU = 0;
	public static final int SOFTMAX = 1;
	
	int previous = 0;
	
	@Override
	public void addLayer(int type, int... layerDescriptor) {
		layers.add(new FullyConnected(type, previous, layerDescriptor[0], type));
		previous = layerDescriptor[0];
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
