package neuralnet.network;

import neuralnet.Learnable;

/**
* @author Muti Kara
*/
public interface Forwardable extends Learnable {
	public Object forwardPropagation(Object inputs);
	public Object backPropagation(Object errors);
}
