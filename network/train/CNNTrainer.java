package nnet.network.train;

import nnet.algebra.Matrix;
import nnet.network.net.ANN;
import nnet.network.net.CNN;

/**
* CNNTrainer
* @author Muti Kara
*/
public class CNNTrainer extends Trainer {
	CNN[] gen;
	CNN father = new CNN();
	CNN mother = new CNN();
	double[] errors;
	int genSize;
	
	ANNTrainer miniTrainer;
	int miniStochastic;
	int miniEpoch;
	double miniRate;
	double miniMomentum;
	double miniRandomization;
	
	public CNNTrainer(CNN net, Matrix[] inputs, Matrix[] answers, double splitRatio) {
		super(net, inputs, answers, splitRatio);
		for (int i = 0; i < net.size(); i++) {
			father.addLayer(net.getLayer(i).getType(), net.getLayer(i).information);
			mother.addLayer(net.getLayer(i).getType(), net.getLayer(i).information);
		}
	}
	
	public void setMiniTrainer(int miniEpoch, int miniStochastic, double miniRate, double miniMomentum, double miniRandomization) {
		this.miniEpoch = miniEpoch;
		this.miniStochastic = miniStochastic;
		this.miniRate = miniRate;
		this.miniMomentum = miniMomentum;
		this.miniRandomization = miniRandomization;
	}

	@Override
	public void preStochastic(int atEpoch, int genSize, double mutation) {
		System.out.println("EPOCH: " + atEpoch);
		this.genSize = genSize;
		errors = new double[genSize];
		if (atEpoch == 0) {
			gen = new CNN[genSize];
			for (int k = 0; k < genSize; k++) {
				gen[k] = new CNN();
				for (int i = 0; i < net.size(); i++) {
					gen[k].addLayer(net.getLayer(i).getType(), net.getLayer(i).information);
				}
				gen[k].randomize(mutation);
			}
		} else {
			Matrix[][][] breed = new Matrix[net.size()][][];
			for (int i = 0; i < net.size(); i++) {
				breed[i] = new Matrix[net.getLayer(i).size()][];
				for (int j = 0; j < net.getLayer(i).size(); j++) {
					breed[i][j] = Matrix.breed(father.getLayer(i).getParameters(j), mother.getLayer(i).getParameters(j), genSize, mutation);
				}
			}
			for (int k = 0; k < genSize; k++) {
				for (int i = 0; i < net.size(); i++) {
					for (int j = 0; j < net.getLayer(i).size(); j++) {
						gen[k].getLayer(i).setParameters(j, breed[i][j][k]);
					}
				}
			}
		}
	}
	
	@Override
	public void stochastic(int at) {
		ANN ann = new ANN();
		Matrix[] annInput = new Matrix[miniStochastic];
		Matrix[] annAnswr = new Matrix[miniStochastic];
		for (int i = 0; i < miniStochastic; i++) {
			Matrix[] data = nextData();
			annInput[i] = gen[at].forwardPropagation( data[0] );
			annAnswr[i] = data[1];
		}
		ann.addLayer(ANN.INPUT, annInput[0].getRow());
		ann.addLayer(ANN.SOFTMAX, answers[0].getRow());
		ann.randomize(miniRandomization);
		miniTrainer = new ANNTrainer(ann, annInput, annAnswr, 1);
		miniTrainer.train(miniEpoch, miniStochastic, miniRate, miniMomentum);
		errors[at] = miniTrainer.getError();
		if (Double.isNaN(errors[at])) {
			errors[at] = Double.MAX_VALUE;
		}
	}

	@Override
	public void postStochastic(double rate, double momentum) {
		int fr = 0;
		int sd = 0;
		for (int i = 0; i < genSize; i++) {
			if (errors[i] <= errors[sd]) {
				if (errors[i] < errors[fr]) {
					sd = fr;
					fr = i;
				} else {
					sd = i;
				}
			}
		}
		double avg = 0;
		for (int i = 0; i < genSize; i++) {
			System.out.print(errors[i] + ", ");
			avg += errors[i];
		}
		System.out.println();
		System.out.println("First: " + fr + ": " + errors[fr]);
		System.out.println("Second: " + sd + ": " + errors[sd]);
		System.out.println("Average: " + avg / genSize);
		for (int i = 0; i < net.size(); i++) {
			for (int j = 0; j < net.getLayer(i).size(); j++) {
				father.getLayer(i).setParameters(j, gen[fr].getLayer(i).getParameters(j));
				mother.getLayer(i).setParameters(j, gen[sd].getLayer(i).getParameters(j));
			}
		}
		if (fr == sd) {
			mother = (CNN) net;
		}
		net = father;
	}

}
