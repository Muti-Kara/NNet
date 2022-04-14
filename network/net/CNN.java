package neuralnet.network.net;

import neuralnet.matrix.Matrix;
import neuralnet.network.layer.Convolutional;
import neuralnet.network.layer.Pooling;

/**
* @author Muti Kara
*/
public class CNN extends Network {
	final static int CONVOLUTIONAL_LAYER = 1;
	final static int POOLING_LAYER = 2;
	
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

	@Override
	public void trainingEpochStep(Matrix input, Matrix answer) {
		// TODO Auto-generated method stub
	}
	
}
