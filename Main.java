import java.io.FileWriter;
import java.io.IOException;

import algebra.NetworkOrganizer;
import preproccess.images.InputImage;
import training.CNNTrainer;


/**
* Main
*/
public class Main {
	
	public static void main(String[] args) throws IOException {
		NetworkOrganizer.addConvolutionalLayer(2, 3, 2);
		NetworkOrganizer.addConvolutionalLayer(2, 3, 2);
		NetworkOrganizer.addFullyConnectedLayer(256);
		NetworkOrganizer.addFullyConnectedLayer(3);
		
		NetworkOrganizer.setEpoch(150);
		NetworkOrganizer.setStochastic(100);
		
		NetworkOrganizer.setCnnEpoch(10);
		NetworkOrganizer.setCnnGeneration(18);
		
		NetworkOrganizer.setImageSize(35);
		
		NetworkOrganizer.setLearningRate(0.02);
		NetworkOrganizer.setMomentumFactor(0.01);
		NetworkOrganizer.setCnnLearningRate(0.01);
		
		NetworkOrganizer.setKernelRandomization(0.1);
		NetworkOrganizer.setRandomization(0.01);
		
		InputImage dataImg = new InputImage("dataset", 'A', 'C');
		NetworkOrganizer.dataSize = dataImg.readFolder();
		
		CNNTrainer trainer = new CNNTrainer(dataImg);
		trainer.train();
		
		//try (FileWriter writer = new FileWriter("RR")) {
		//	writer.write(trainer.getBest().toString());
		//}
		
		// NeuralNetwork network = Reader.readNN();
		// for(int index = 0; index < NetworkParameters.testSize; index++){
		//  	String str = network.classify( dataImg.getTests(index) );
		//  	System.out.println(dataImg.getExpected(index) + str);
		// }
		
	}
	
}




