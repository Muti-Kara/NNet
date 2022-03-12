package preproccess.images;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import algebra.NetworkParameters;
import algebra.matrix.Matrix;

/**
* InputImage
* This class reads a folder and gets the input and answer.
* @author Muti Kara
*/
public class InputImage {
	int numOfData;
	String folderName = NetworkParameters.readDir + "/images/data/";
	
	Matrix[] inputs;
	Matrix answers;
	
	/**
	* These folders consists of 26 sub folders each named with an upper case english letter.
	* @throws IOException
	 */
	public InputImage(String folderName) throws IOException{
		this.folderName += folderName + "/";
	}
	
	public int readFolder() throws IOException{
		ArrayList<String> fileNames = new ArrayList<>();
		for(char character = 'A'; character <= 'Z'; character++){
			File[] files = new File(folderName + character).listFiles();
			for(File file : files){
				if(file.isFile())
					fileNames.add(file.getName() + " " + character);
			}
		}
		inputs = new Matrix[fileNames.size()];
		answers = new Matrix(fileNames.size(), NetworkParameters.structure[NetworkParameters.structure.length - 1]);
		for(int i = 0; i < fileNames.size(); i++){
			String file = fileNames.get(i).substring(0, fileNames.get(i).length() - 2);
			char character = fileNames.get(i).charAt(fileNames.get(i).length() - 1);
			ImageBuffer img = new ImageBuffer(folderName + character + "/" + file);
			img.resize();
			img.turnBlackAndWhite();
			img.maximizeContrast();
			inputs[i] = img.getMatrix();
			answers.set(i, character - 'A', 1);
		}
		return fileNames.size();
	}
	
	/**
	* 
	* @param index
	* @return index th input
	 */
	public Matrix getInput(int index) {
		return inputs[index];
	}
	
	/**
	* 
	* @param index
	* @return index th answer
	 */
	public char getAnswer(int index) {
		for(char i = 'A'; i <= 'Z'; i++)
			if(answers.get(index, i - 'A') == 1)
				return i;
		return '!';
	}
	
	/**
	* 
	* @return all inputs
	 */
	public Matrix[] getInputs() {
		return inputs;
	}
	
	/**
	* 
	* @return all answers
	 */
	public Matrix getAnswers() {
		return answers;
	}
	
}
