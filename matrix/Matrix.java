package neuralnet.matrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

import neuralnet.Learnable;

/**
 * @author Muti Kara
 * */
public class Matrix implements Learnable {
	Random rand = new Random();
	double[][] matrix;
	int col, row;
	
	public static Matrix flatten(Matrix[] matArray){
		if (matArray.length == 1)
			return matArray[0];
		
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
	
	public Matrix(int row, int col) {
		matrix = new double[row][col];
		this.col = col;
		this.row = row;		
	}
	
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
		row = matrix.length;
		col = matrix[0].length;
	}
	
	public void set(int row, int col, double value){
		matrix[row][col] = value;
	}
	
	public double get(int row, int col){
		return matrix[row][col];
	}
	
	public int getCol() {
		return col;
	}

	public int getRow() {
		return row;
	}
	
	public Matrix abs() {
		for(int r = 0; r < row; r++)
			for(int c = 0; c < col; c++)
				matrix[r][c] = Math.abs(matrix[r][c]);
		return this;
	}
	
	public Matrix randomize(double d) {
		for(int r = 0; r < row; r++)
			for(int c = 0; c < col; c++)
				matrix[r][c] = rand.nextGaussian() * d;
		return this;
	}
	
	public Matrix createClone() {
		Matrix C = new Matrix(row, col);
		for(int r = 0; r < row; r++) {
			C.matrix[r] = matrix[r].clone();
		}
		return C;
	}
	
	public Matrix getVector(int row) {
		Matrix C = new Matrix(col, 1);
		for(int r = 0; r < C.row; r++) {
			C.matrix[r][0] = matrix[row][r];
		}
		return C;
	}
	
	public Matrix dot(Matrix B) {
		if(this.col != B.row) {
			System.out.println("Matrix dot error. Size Mismatch :" + col + " " + B.row);
			return null;
		}
		Matrix C = new Matrix(this.row, B.col);
		for(int crow = 0; crow < this.row; crow++) {
			for(int ccol = 0; ccol < B.col; ccol++) {
				double c = 0;
				for(int sum = 0; sum < this.col; sum++) {
					c += matrix[crow][sum] * B.matrix[sum][ccol];
				}
				C.matrix[crow][ccol] = c;
			}
		}
		return C;
	}
	
	public Matrix sum(Matrix B) {
		if(!(row == B.row && col == B.col)) {
			System.out.println("Matrix sum error. Size Mismatch.");
			System.out.println("Row1: " + row + "\tRow2: " + B.row);
			System.out.println("Col1: " + col + "\tCol2: " + B.col);
			return null;
		}
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] += B.matrix[r][c];
			}
		}
		return this;
	}
	
	public Matrix sub(Matrix B) {
		if(!(row == B.row && col == B.col)) {
			System.out.println("Matrix sub error. Size Mismatch.");
			return null;
		}
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] -= B.matrix[r][c];
			}
		}
		return this;
	}
	
	public Matrix scalarProd(double b) {
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] *= b;
			}
		}
		return this;
	}
	
	public Matrix scalarSum(double b) {
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				matrix[r][c] += b;
			}
		}
		return this;
	}
	
	public Matrix transpose() {
		Matrix C = new Matrix(col, row);
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++) {
				C.matrix[c][r] = matrix[r][c];
			}
		}
		return C;
	}
	
	public Matrix relu(){
		for(int r = 0; r < getRow(); r++) {
			for(int c = 0; c < getCol(); c++) {
				matrix[r][c] = Math.max(0, matrix[r][c]); 
			}
		}
		return this;
	}
	
	public Matrix d_relu() {
		for(int r = 0; r < getRow(); r++) {
			for(int c = 0; c < getCol(); c++) {
				matrix[r][c]  = (matrix[r][c] <= 0)? 0 : 1;
			}
		}
		return this;
	}
	
	public Matrix softmax(){
		double sum = 0;
		for(int r = 0; r < row; r++){
			matrix[r][0] = Math.exp(matrix[r][0]);
			sum += matrix[r][0];
		}
		for(int r = 0; r < row; r++){
			matrix[r][0] /= sum;
		}
		return this;
	}
	
	public Matrix mutate(int r, double rate) {
		Matrix child = new Matrix(row, col);
		for(int c = 0; c < child.getCol(); c++){
			child.set(r, c, matrix[r][c] + rand.nextGaussian() * rate);
		}
		return child;
	}
	
	@Override
	public String toString() {
		String str = "";
		for(int r = 0; r < row; r++) {
			for(int c = 0; c < col; c++)
				str += matrix[r][c] + " ";
			str += "\n";
		}
		return str;
	}

	@Override
	public void read(Scanner in) {
		for(int r = 0; r < row; r++)
			for(int c = 0; c < col; c++)
				matrix[r][c] = in.nextDouble();
	}
	
	@Override
	public void write(FileWriter out) {
		try {
			out.write( this.toString() );
		} catch (IOException e) {
			System.out.println(e.fillInStackTrace());
		}
	}

}
