package nnet.network;

import nnet.Learnable;

/**
* An interface for forward propagatable objects
* @author Muti Kara
*/
public interface Forwardable<T> extends Learnable {
	public T forwardPropagation(T inputs);
}
