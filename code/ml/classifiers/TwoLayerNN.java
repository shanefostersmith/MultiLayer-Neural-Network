// Shane Foste-Smith, CS158, Assignment 8

package ml.classifiers;

import ml.data.*;
import ml.utils.*;
import java.util.Map;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.lang.Math;
import java.util.Random;

public class TwoLayerNN implements Classifier {
	
	protected static int hNodes; // number hidden nodes w/o bias
	protected double n = 0.1;
	protected int iterations = 200;
	public boolean tanhActivation = true;
	
	protected ArrayList<Node> featureNodes;
	protected ArrayList<Node> hiddenNodes;
	
	protected int dataSetFeatures; //total features in data set w/o bias
	protected HashMap<Integer, String> featureMap;
	
	/**
	 * Constructor
	 * @param number of hidden nodes
	 */
	public TwoLayerNN(int hNodes) {
		this.hNodes = hNodes;
		featureNodes = new ArrayList<Node>();
		hiddenNodes = new ArrayList<Node>();
		
	}
	
	/**
	 * Set learning rate for classifier
	 * 
	 * @param learning rate
	 */
	public void setEta(double learningRate) {
		this.n = learningRate;
	}
	
	/**
	 * Set number of iterations for classifier
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}
	
	/**
	 * Train this classifier based on the data set
	 * 
	 * @param data set
	 */
	public void train(DataSet data) {
		DataSet dataWBias = data.getCopyWithBias();
		dataSetFeatures = data.getAllFeatureIndices().size();
		featureMap = data.getFeatureMap();
		
		
		// clear node arrays
		if (!featureNodes.isEmpty()) { featureNodes.clear();}
		if (!hiddenNodes.isEmpty()) { hiddenNodes.clear();}
		
		// Populate input node list
		for (int i = 0; i < dataSetFeatures; i++) {
			Node inputnode = new Node(i, false);
			featureNodes.add(inputnode);
		}
		// Add bias node to input node list
		int biasIndex = dataSetFeatures;
		Node biasnode = new Node(biasIndex,false);
		biasnode.setNodeValue(1.0);
		featureNodes.add(biasnode);
		
		// Populate hidden node list
		for (int i = 0; i < hNodes; i++) {
			Node hiddennode = new Node(i, true);
			hiddenNodes.add(hiddennode);
		}
		// Add bias node to hidden node list
		int hBiasIndex = hNodes;
		Node hBiasnode = new Node(hBiasIndex, true);
		hBiasnode.setNodeValue(1.0);
		hiddenNodes.add(hBiasnode);
		
		 //Set random weight vectors to all nodes
		Random randomGenerator = new Random();
		for (int j = 0; j <= dataSetFeatures; j++) {
			Node currentNode = featureNodes.get(j);
			for (int k = 0; k < hNodes; k++) {
				double randNum = -0.1 + (0.2 * randomGenerator.nextDouble());
				currentNode.addWeight(randNum);
			}
		}
		for (int k = 0; k <= hNodes; k++) {
			Node currentNode = hiddenNodes.get(k);
			double randNum = -0.1 + (0.2 * randomGenerator.nextDouble());
			currentNode.addWeight(randNum);
		}
		
		//buildNetwork1();
		for (int iter = 0; iter < iterations; iter++) {
			
		Collections.shuffle(dataWBias.getData());

		for (Example e : dataWBias.getData()) {
			double[] newVweights = new double[hNodes + 1];
			double[][] newWweights = new double[dataSetFeatures+1][hNodes];
			
			double outputNoAct = calculateOutputNoA(e);
			double output = activation(outputNoAct);
			double label = e.getLabel();
			if (!tanhActivation) {
				if (label == -1) {label = 0.0;} // change -1 label if sigmoid activation
			}
			double loss = label - output;

			// update output layer
			double recursiveProduct = (loss) * ddxActivation(outputNoAct);
			for (int v = 0; v <= hNodes; v++) {
				Node currentNode = hiddenNodes.get(v);
				double weight = currentNode.getWeight(0);
			
				// calculate new weights;
				double newWeight = weight + n * currentNode.nodeValue * recursiveProduct;
				newVweights[v] = newWeight;
			}

			// find new input node weights
			for (int j = 0; j <= dataSetFeatures; j++) {
				Node currentNode = featureNodes.get(j); 
				
				for (int k = 0; k < hNodes; k++) {
					double oldWeight = currentNode.getWeight(k);
					double wSlope = ddxActivation(calculateWSum(k));
					double vk = hiddenNodes.get(k).getWeight(0);
					double newWeight = oldWeight + n * currentNode.nodeValue * wSlope * vk * recursiveProduct;
					newWweights[j][k] = newWeight;
				}
			}
			
			// update weights in nodes
			for (int j = 0; j <= dataSetFeatures; j++) {
				Node featNode = featureNodes.get(j);
				for (int k = 0; k < hNodes; k++) {
					featNode.updateWeight(k, newWweights[j][k]);
				}		
			}
			for (int k = 0; k <= hNodes; k++) {
				Node hiddenNode = hiddenNodes.get(k);
				hiddenNode.updateWeight(0, newVweights[k]);
			}
		}
		} // end iteration
	}

	/**
	 * Calculate sum of weights going inputed into a hidden node
	 * 
	 * @param hidden nodes index
	 */
	private double calculateWSum(int hNodeNum) {
		double sum = 0;
		for (int w = 0; w <= dataSetFeatures; w++) {
			Node currentNode = featureNodes.get(w);
			double weight = currentNode.getWeight(hNodeNum);
			sum += weight * currentNode.nodeValue;
		}
		return sum;
	}
	
	/**
	 * Calculate input value to output node an example
	 * going through the current network
	 * 
	 * @param hidden nodes index
	 */
	private double calculateOutputNoA (Example e) {
		
		double[] hk = new double[hNodes];
		
		// iterate through input nodes
		for (int feat = 0; feat <= dataSetFeatures; feat++) {
			double featureValue = e.getFeature(feat);
			Node inputNode = featureNodes.get(feat);
			if (feat != dataSetFeatures) {
				inputNode.setNodeValue(featureValue);
			}
			
			// iterate through weight vectors
			for (int k = 0; k < hNodes; k++) {
				double weight = inputNode.getWeight(k);
				hk[k] += featureValue*weight;
			}
		}
		
		// apply activation function on sums
		for (int k = 0; k < hNodes; k++) {
			hk[k] = activation(hk[k]);
			hiddenNodes.get(k).setNodeValue(hk[k]);
		}
		
		// Calculate output
		double output = 0;
		for (int k = 0; k <= hNodes; k++) {
			Node hiddennode = hiddenNodes.get(k);
			output += hiddennode.nodeValue * hiddennode.getWeight(0);
		}
		
		return output;
	}
	
	/**
	 * Classify the example.  Should only be called *after* train has been called.
	 * 
	 * @param example
	 * @return the class label predicted by the classifier for this example
	 */
	public double classify(Example example) {
		
		// Get new example with bias feature
		DataSet newExSet0 = new DataSet(featureMap);
		DataSet newExSet1= newExSet0.getCopyWithBias();
		Example e = newExSet1.addBiasFeature(example); 
		
		double value = activation(calculateOutputNoA(e));
		if (tanhActivation) {
			 return (value > 0.0) ? 1.0 : -1.0;
		} else {
			return(value >= 0.5) ? 1.0 : 0;
		}
		
	}
	
	/**
	 * Calculate confidence/out of an example
	 * in the current network
	 * 
	 * @param hidden nodes index
	 */
	public double confidence(Example example) {
	
		// Get new example with bias feature
		DataSet newExSet0 = new DataSet(featureMap);
		DataSet newExSet1= newExSet0.getCopyWithBias();
		Example e = newExSet1.addBiasFeature(example);
		
		double value = activation(calculateOutputNoA(e));
			if (value > 0.0) {
				return value;
			} else {
				return -value;
			}
	
	}
	
	/**
	 * Current network to string
	 */
	public void networkToString() {
		
		for(int j = 0; j < dataSetFeatures; j++) {
			System.out.println("Feature " + (j+1) + ": " + featureNodes.get(j).nodeValue);
			for (int k = 0; k < hNodes; k++) {
				System.out.println("	Weight "+ (k+1) + ": " + featureNodes.get(j).getWeight(k));
			}
		}
		System.out.println("Feature Bias Value: " + featureNodes.get(dataSetFeatures).nodeValue);
		for (int k = 0; k < hNodes; k++) {
			
			System.out.println("Feature Bias Weight " + (k+1) +": " + featureNodes.get(dataSetFeatures).getWeight(k));
		}
		System.out.println("");
		for (int k = 0; k < hNodes; k++) {
			System.out.println("Hidden Node " + (k+1) + " Weight: " + hiddenNodes.get(k).getWeight(0));
		}
		System.out.println("Hidden Node Bias Value: " + hiddenNodes.get(hNodes).nodeValue);
		System.out.println("Hidden Node Bias Weight: " + hiddenNodes.get(hNodes).getWeight(0));
		
	}
	
	/**
	 * Set activation function to tanh
	 */
	public void setTanhActivation() {
		this.tanhActivation = true;
	}
	
	/**
	 * Set activation function to sigmoid
	 */
	public void setSigmoidActivation() {
		this.tanhActivation = false;
	}
	
	/**
	 * sigmoid function calculation
	 */
	private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
	
	/**
	 *  Activation function calculations
	 */
	private double activation(double x) {
		if (tanhActivation) {
			return Math.tanh(x);
		} else {
			return sigmoid(x);
		}
	}
	
	/**
	 *  Derivative of activation functions
	 */
	private double ddxActivation(double outputSum) {
		if (tanhActivation) {
			return Math.pow(1.0 / Math.cosh(outputSum), 2);
		} else {
			return sigmoid(outputSum) * (1 - sigmoid(outputSum));
		}
	}
	
	/**
	 *  Build network from Figure 1
	 */
	private void buildNetwork1() {
		featureNodes.get(0).addWeight(-0.7);
		featureNodes.get(0).addWeight(0.03);
				
		featureNodes.get(1).addWeight(1.6);
		featureNodes.get(1).addWeight(0.6);
				
		featureNodes.get(2).addWeight(-1.8);
		featureNodes.get(2).addWeight(-1.4);
				
		hiddenNodes.get(0).addWeight(-1.1);
		hiddenNodes.get(1).addWeight(-0.6);
		hiddenNodes.get(2).addWeight(1.8);
	}
	/**
	 * A class representing a node with an internal value
	 * weights directed out of it
	 */
	private static class Node {
		public int nodeNum;
		public double nodeValue;
		
		public boolean hiddenNode;
		public ArrayList<Double> vectors;
		
		
		/**
		 *  Node Constructor
		 *  
		 *  @param node index
		 *  @param hiddenNode: if this node a hidden node
		 */
		public Node(int nodeNum, boolean hiddenNode) {
			this.nodeNum = nodeNum;
			this.hiddenNode = hiddenNode;
			this.nodeValue = 0;
			this.vectors = new ArrayList<Double>();
		}
		
		/**
		 *  Set the value of the node
		 */
		public void setNodeValue (double value) {
			this.nodeValue = value;
		}
		
		/**
		 *  Add weight vector
		 */
		public void addWeight(double weight) {
			vectors.add(weight);
		}
		
		/**
		 *  Add update weight vector going to specified hidden node
		 *  (or output node)
		 */
		public void updateWeight(int toHNode, double weight) {
			if ( ( (toHNode > hNodes) && !hiddenNode ) || ( (toHNode > 0) && hiddenNode ) ) {
				System.out.println("Error, that weight vector doesn't exist");
			} else {
				vectors.set(toHNode, weight);
			}
			
		}
		
		/**
		 *  Retrieve weight vector going to specified hidden node
		 *  or output node
		 */
		public double getWeight(int toHNode) {
			return vectors.get(toHNode);
		}
		
		// New equality definition for a Node
		@Override
		public boolean equals(Object o) {
			Node node = (Node) o;
			return (nodeNum == node.nodeNum);
		}
	}
}
	
