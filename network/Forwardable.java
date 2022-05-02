package nnet.network;

import nnet.Learnable;

/**
* @author Muti Kara
*/
public interface Forwardable<T> extends Learnable {
	public T forwardPropagation(T inputs);
}
