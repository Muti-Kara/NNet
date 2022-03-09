package preproccess;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import algebra.HyperParameters;
import algebra.Matrix;

/**
* InputImage
* This class reads dataset folder and gets the training and test data.
* @author Muti Kara
*/
public class InputImage {
	int dataSize;
	int testSize;
	String projectDir = HyperParameters.projectDir;
	
	Matrix[] inputs;
	Matrix[] tests;
	Matrix answers;
	Matrix expected;
	
	/**
	* Scans projectDir/dataset and projectDir/tests folders.
	* These folders consists of 26 sub folders each named with an upper case english letter.
	* @throws IOException
	 */
	public InputImage() throws IOException{
		HyperParameters.DATA_SIZE = readFolder("dataset", true);
		HyperParameters.TEST_SIZE = readFolder("tests", false);
	}
	
	public int readFolder(String folderName, boolean isData) throws IOException{
		ArrayList<String> fileNames = new ArrayList<>();
		for(char character = 'A'; character <= 'Z'; character++){
			File[] files = new File(projectDir + "/" + folderName + "/" + character).listFiles();
			for(File file : files){
				if(file.isFile())
					fileNames.add(file.getName() + " " + character);
			}
		}
		if(isData){
			inputs = new Matrix[fileNames.size()];
			answers = new Matrix(fileNames.size(), HyperParameters.structure[HyperParameters.structure.length - 1]);
		}else{
			tests = new Matrix[fileNames.size()];
			expected = new Matrix(fileNames.size(), HyperParameters.structure[HyperParameters.structure.length - 1]);
		}
		for(int i = 0; i < fileNames.size(); i++){
			String file = fileNames.get(i).substring(0, fileNames.get(i).length() - 2);
			char character = fileNames.get(i).charAt(fileNames.get(i).length() - 1);
			ImageBuffer img = new ImageBuffer(projectDir + "/" + folderName + "/" + character + "/" + file);
			img.resize();
			img.turnBlackAndWhite();
			img.maximizeContrast();
			if(isData){
				inputs[i] = img.getMatrix();
				scaleInputs(inputs[i]);
				answers.set(i, character - 'A', 1);
			}else{
				tests[i] = img.getMatrix();
				scaleInputs(tests[i]);
				expected.set(i, character - 'A', 1);
			}
		}
		return fileNames.size();
	}
	
	/**
	* Scales input in a binary way.
	* @param matrix
	 */
	public void scaleInputs(Matrix matrix){
		for(int r = 0; r < matrix.getRow(); r++)
			for(int c = 0; c < matrix.getCol(); c++)
				matrix.set(r, c, (matrix.get(r, c) == 0)? 0 : 1);
	}
	
	public Matrix getInputs(int index) {
		return inputs[index];
	}
	
	public Matrix getTests(int index) {
		return tests[index];
	}
	
	public Matrix getAnswers() {
		return answers;
	}
	
	public Matrix getExpected() {
		return expected;
	}
	
}
