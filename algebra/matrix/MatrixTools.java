package neuralnet.algebra.matrix;

import java.util.Random;

/**
* MatrixTools
* @author Muti Kara
*/
public class MatrixTools {
	static Random rand = new Random();
	
	/**
	* Applies ReLU activation function to all of the parameters.
	* @param matrix
	* @return matrix itself
	 */
	public static Matrix relu(Matrix matrix){
		for(int r = 0; r < matrix.getRow(); r++) {
			for(int c = 0; c < matrix.getCol(); c++) {
				matrix.set(r, c, Math.max(0, matrix.get(r, c))); 
			}
		}
		return matrix;
	}
	
	/**
	* Applies ReLU activation function's derivative to all of the parameters.
	* @param matrix
	* @return matrix itself
	 */
	public static Matrix d_relu(Matrix matrix) {
		for(int r = 0; r < matrix.getRow(); r++) {
			for(int c = 0; c < matrix.getCol(); c++) {
				matrix.set(r, c, (matrix.get(r, c) <= 0)? 0 : 1);
			}
		}
		return matrix;
	}
	
	/**
	* Applies Softmax function to vector.
	* @param vector
	* @return vector itself
	 */
	public static Matrix softmax(Matrix vector){
		double sum = 0;
		for(int r = 0; r < vector.getRow(); r++){
			vector.set(r, 0, Math.exp(vector.get(r, 0)));
			sum += vector.get(r, 0);
		}
		for(int r = 0; r < vector.getRow(); r++){
			vector.set(r, 0, vector.get(r, 0) / sum);
		}
		return vector;
	}
	
	/**
	* Converts the matrix array to a vector.
	* @param matArray
	* @return flattened matrix array
	 */
	public static Matrix flatten(Matrix[] matArray){
		Matrix vector = new Matrix(matArray.length * matArray[0].getRow() * matArray[0].getCol(), 1);
		int k = 0;
		for(int i = 0; i < matArray.length; i++){
			for(int r = 0; r < matArray[i].getRow(); r++){
				for(int c = 0; c < matArray[i].getCol(); c++){
					vector.set(k++, 0, matArray[i].get(r, c));
				}
			}
		}
		return vector;
	}
	
	/**
	* Generates a new matrix by changing a row of parent matrix
	* @param parent
	* @param row
	* @return child matrix
	*/
	public static Matrix generate(Matrix parent, int r, double rate) {
		Matrix child = new Matrix(parent.getRow(), parent.getCol());
		for(int c = 0; c < child.getCol(); c++){
			child.set(r, c, parent.get(r, c) + rand.nextGaussian() * rate);
		}
		return child;
	}
	
}
