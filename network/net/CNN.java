package neuralnet.network.net;

import neuralnet.network.layer.Convolutional;
import neuralnet.network.layer.Pooling;

/**
* A simple convolutional network
* @author Muti Kara
*/
public class CNN extends Net {
	public final static int CONVOLUTION = 0;
	public final static int MAX_POOL = 1;
	
	@Override
	public void addLayer(int type, int ... layerDescriptor) {
		switch (type) {
			case CONVOLUTION:
				layers.add(new Convolutional(type, layerDescriptor));
				break;
			case MAX_POOL:
				layers.add(new Pooling(type, layerDescriptor));
				break;
		}
	}
	
}
