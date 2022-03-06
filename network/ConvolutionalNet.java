package network;

import java.io.IOException;

import algebra.HyperParameters;
import algebra.Matrix;
import algebra.MatrixTools;
import layer.Convolutional;
import layer.Pooling;

/**
* A simple convolutional network
* @author Muti Kara
*/
public class ConvolutionalNet {
	int[] convolutional = HyperParameters.convolutional;
	int[] kernel = HyperParameters.kernel;
	int[] pool = HyperParameters.pool;
	Convolutional[] convLayers = new Convolutional[convolutional.length];
	Pooling[] poolLayers = new Pooling[pool.length];
	
	public ConvolutionalNet() throws IOException{
		for(int i = 0; i < convLayers.length; i++){
			convLayers[i] = new Convolutional(convolutional[i], kernel[i]);
			poolLayers[i] = new Pooling(pool[i]);
		}
	}
	
	public Matrix forwardPropagation(Matrix input) {
		Matrix[] preOutput = new Matrix[1];
		preOutput[0] = input;
		
		for(int i = 0; i < convolutional.length; i++){
			preOutput = convLayers[i].goForward(preOutput);
			preOutput = poolLayers[i].goForward(preOutput);
		}
		
		return MatrixTools.flatten(preOutput);
	}
	
	public void trainCNN() {
		
	}
	
}
