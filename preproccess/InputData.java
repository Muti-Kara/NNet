package preproccess;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import algebra.HyperParameters;
import algebra.Matrix;

/**
* InputData
*/
public class InputData {
	int[] structure = HyperParameters.structure;
	int dataSize = HyperParameters.DATA_SIZE;
	int testSize = HyperParameters.TEST_SIZE;
	
	Matrix inputs = new Matrix(dataSize, structure[0]);
	Matrix answers = new Matrix(dataSize, structure[structure.length - 1]);
	
	Matrix asked = new Matrix(testSize, structure[0]);
	Matrix expected = new Matrix(testSize, structure[structure.length - 1]);
	
	public InputData(String fileName) throws FileNotFoundException{
		Scanner scan = new Scanner(new File(HyperParameters.projectDir + fileName));
		
		for(int i = 0; i < dataSize; i++){
			for(int j = 0; j < structure[0]; j++)
				inputs.set(i, j, scan.nextDouble());
			for(int j = 0; j < structure[structure.length - 1]; j++)
				answers.set(i, j, scan.nextDouble());
		}
		
		for(int i = 0; i < testSize; i++){
			for(int j = 0; j < structure[0]; j++)
				asked.set(i, j, scan.nextDouble());
			for(int j = 0; j < structure[structure.length - 1]; j++)
				expected.set(i, j, scan.nextDouble());
		}
	}
	
	public Matrix getInputs() {
		return inputs;
	}
	
	public Matrix getAnswers() {
		return answers;
	}

	public Matrix getAsked() {
		return asked;
	}

	public Matrix getExpected() {
		return expected;
	}

}
