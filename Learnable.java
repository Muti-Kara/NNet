package nnet;

import java.io.FileWriter;
import java.util.Scanner;

/**
* Learnable
* An interface for learnable objects
* @author Muti Kara
*/
public interface Learnable {
	public void read(Scanner in);
	public void write(FileWriter out);
}
