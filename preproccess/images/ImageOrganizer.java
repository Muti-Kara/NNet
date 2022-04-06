package neuralnet.preproccess.images;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import neuralnet.algebra.NetworkOrganizer;
import neuralnet.algebra.matrix.Matrix;

/**
* InputImage
* This class reads a folder and gets the input and answer.
* @author Muti Kara
*/
public class ImageOrganizer {
	char start, end;
	int numOfData;
	String folderName = NetworkOrganizer.readDir + "/images/data/";
	
	Matrix[] inputs;
	Matrix answers;
	
	/**
	* These folders consists of 26 sub folders each named with an upper case english letter.
	* @throws IOException
	 */
	public ImageOrganizer(String folderName, char start, char end) throws IOException{
		this.start = start;
		this.end = end;
		this.folderName += folderName + "/";
	}
	
	public int readFolder() throws IOException{
		ArrayList<String> fileNames = new ArrayList<>();
		for(char character = start; character <= end; character++){
			System.out.println("Folder loaded: " + folderName + character);
			File[] files = new File(folderName + character).listFiles();
			for(File file : files){
				if(file.isFile())
					fileNames.add(file.getName() + " " + character);
			}
		}
		inputs = new Matrix[fileNames.size()];
		answers = new Matrix(fileNames.size(), NetworkOrganizer.structure[NetworkOrganizer.structure.length - 1]);
		for(int i = 0; i < fileNames.size(); i++){
			String file = fileNames.get(i).substring(0, fileNames.get(i).length() - 2);
			char character = fileNames.get(i).charAt(fileNames.get(i).length() - 1);
			// TODO read image and convert it to a matrix
			// GrayBuffer img = new GrayBuffer(folderName + character + "/" + file);
			// img.resize();
			// img.turnBlackAndWhite();
			// img.maximizeContrast();
			// img.write(NetworkOrganizer.readDir + "/images/check/" + character + "/" + file);
			// inputs[i] = img.getMatrix();
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
		for(char i = start; i <= end; i++)
			if(answers.get(index, i - start) == 1)
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
