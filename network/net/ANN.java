package neuralnet.network.net;

import neuralnet.network.layer.FullyConnected;

/**
 * An artificial neural network
 * @author Muti Kara
 * */
public class ANN extends Network {
	int previous = 0;

	/**
	* adds a new layer
	* @param size of new layer
	* @param activation type
	* @return this network
	*/
	@Override
	public Network addLayer(int ... layerDescription) {
		if(layerDescription.length != 2) {
			System.out.println("Wrong parameter format");
			return null;
		}
		layers.add(new FullyConnected(previous, layerDescription[0], layerDescription[1]));
		previous = layerDescription[0];
		return this;
	}
	
}
