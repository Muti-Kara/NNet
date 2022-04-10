package neuralnet.network.net;

import neuralnet.network.layer.Convolutional;
import neuralnet.network.layer.Pooling;

/**
* A simple convolutional network
* @author Muti Kara
*/
public class CNN extends Network {
	final static int CONVOLUTIONAL_LAYER = 1;
	final static int POOLING_LAYER = 2;
	
	/**
	* Adds a new convolutional layer or a pool layer
	* @param descriptor
	* @param type
	* @return this network
	*/
	@Override
	public Network addLayer(int ... layerDescription) {
		switch (layerDescription[layerDescription.length - 1]) {
			case CONVOLUTIONAL_LAYER:
				layers.add( new Convolutional(layerDescription[0], layerDescription[1]) );
				return this;
			
			case POOLING_LAYER:
				layers.add( new Pooling(layerDescription[0]) );
				return this;
			
			default:
				System.out.println("Undefined type of layer.");
				return null;
		}
	}
	
}
