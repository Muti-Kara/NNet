package neuralnet.network.net;

import neuralnet.matrix.Matrix;
import neuralnet.network.layer.FullyConnected;

/**
 * @author Muti Kara
 * */
public class ANN extends Network {
	int previous = 0;

	@Override
	public Network addLayer(int ... layerDescription) {
		if(layerDescription.length != 3) {
			System.out.println("Wrong parameter format");
			return null;
		}
		layers.add(new FullyConnected(previous, layerDescription));
		previous = layerDescription[0];
		return this;
	}

	@Override
	public void trainingEpochStep(Matrix input, Matrix answer) {
		Matrix error = forwardPropagation(input).sub(answer);
		for (int i = layers.size() - 1; i > 0; i--) {
			error = ((FullyConnected) layers.get(i)).backPropagation(error);
			((FullyConnected) layers.get(i)).calculateChanges();
		}
	}
	
}
