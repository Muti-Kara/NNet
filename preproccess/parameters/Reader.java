package neuralnet.preproccess.parameters;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;
import neuralnet.network.NeuralNetwork;
import neuralnet.network.ann.ANN;
import neuralnet.network.ann.layer.FullyConnected;
import neuralnet.network.cnn.CNN;
import neuralnet.network.cnn.layer.Convolutional;
import neuralnet.network.cnn.layer.Pooling;

/**
* Reader class to read pretrained networks.
* @author Muti Kara
*/
public class Reader {
	static Scanner cnnIn;
	static Scanner annIn;
	
	public static NeuralNetwork readNN() throws FileNotFoundException {
		return new NeuralNetwork(readCNN(), readANN());
	}
	
	public static CNN readCNN() throws FileNotFoundException {
		cnnIn = new Scanner(new File(NetworkOrganizer.readDir + "/parameters/cnn/CNN.txt"));
		CNN cnn = new CNN();
		int length = cnnIn.nextInt();
		
		for(int i = 0; i < length; i++){
			cnn.setConvLayer(i, readConvLayer());
			cnn.setPoolLayer(i, new Pooling(cnnIn.nextInt()));
		}
		
		return cnn;
	}
	
	public static Convolutional readConvLayer() {
		int kernelSize = cnnIn.nextInt();
		int numOfKernels = cnnIn.nextInt();
		Convolutional conv = new Convolutional(numOfKernels, kernelSize);
		
		conv.setKernelBiases( readMatrix(numOfKernels, 1, cnnIn) );
		for(int i = 0; i < numOfKernels; i++){
			conv.setKernel(i, readMatrix(kernelSize, kernelSize, cnnIn));
		}
		
		return conv;
	}
	
	public static Matrix readMatrix(int row, int col, Scanner in) {
		Matrix matrix = new Matrix(row, col);
		for(int r = 0; r < row; r++)
			for(int c = 0; c < col; c++)
				matrix.set(r, c, in.nextDouble());
		return matrix;
	}
	
	public static ANN readANN() throws FileNotFoundException {
		annIn = new Scanner(new File(NetworkOrganizer.readDir + "/parameters/ann/ANN.txt"));
		ANN ann = new ANN();
		
		for(int i = 1; i < NetworkOrganizer.structure.length; i++){
			ann.setLayer(i, readFullyConnectedLayer());
		}
		
		return ann;
	}
	
	public static FullyConnected readFullyConnectedLayer() {
		int sizePrev = annIn.nextInt();
		int sizeNext = annIn.nextInt();
		FullyConnected fc = new FullyConnected(sizePrev, sizeNext);
		fc.setWeight( readMatrix(sizeNext, sizePrev, annIn) );
		fc.setBias( readMatrix(sizeNext, 1, annIn) );
		return fc;
	}
	
}
