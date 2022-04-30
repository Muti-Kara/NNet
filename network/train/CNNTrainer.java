package neuralnet.network.train;

import neuralnet.network.net.CNN;
import neuralnet.algebra.Matrix;

/**
* CNNTrainer
* @author Muti Kara
*/
public class CNNTrainer extends Trainer {

	public CNNTrainer(CNN net, Matrix[] inputs, Matrix[] answers, double splitRatio) {
		super(net, inputs, answers, splitRatio);
	}

	@Override
	public void stochastic(int epoch, int stochastic) {
		// TODO Auto-generated method stub
	}

	@Override
	public void apply(double rate, double momentum) {
		// TODO Auto-generated method stub
	}

	@Override
	public void calculateError(Matrix answer) {
		// TODO Auto-generated method stub
	}

}
