import java.io.IOException;

import network.*;
import preproccess.*;
import algebra.*;
import training.*;

/**
* Main
*/
public class Main {
	
	public static void main(String[] args) throws IOException {
		InputImage dataImg = new InputImage();
		CNNTrainer trainer = new CNNTrainer(dataImg);
		trainer.train();
	}
	
}
