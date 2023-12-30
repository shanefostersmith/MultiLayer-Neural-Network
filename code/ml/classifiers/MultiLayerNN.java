package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ml.data.*;

/**
 * Implementation of multi-layer neural network for
 * signle label
 * ,
 */
public class MultiLayerNN implements Classifier {
	
	protected static int hNodes; // number hidden nodes in a layer
	protected int hLayers; // number of hidden layers
	protected double n = 0.00001; // learning rate
	protected int iterations = 50; // iterations over training data
	
	// activation types
	protected boolean reluActivation = true; 
	protected boolean LreluActivation = false;
	protected boolean eluActivation = false;
	protected boolean seluActivation = false;
	
	protected double lambda = 0.0; // regularization lambda
	protected double biasOffset = 1;
	
	// ArrayLists representing the network
	protected ArrayList<Node> featureNodes;
	protected ArrayList<ArrayList<Node>> hiddenNodes;
	protected ArrayList<Node> outputNodes;
	
	// data set information
	protected int dataSetFeatures; //total features in data set w/o bias
	protected int numLabels;
	protected HashMap<Integer, String> featureMap;
	
	final double ALPHA = 1.0; 
	final double LAMBDA = 1.0508; 
	final double SELU_ALPHA = 1.67326; 
	
	/**
	 * Constructor
	 * @param number of hidden nodes
	 */
	public MultiLayerNN(int hLayers, int hNodesPerLayer) {
		this.hNodes = hNodesPerLayer;
		this.hLayers = hLayers;
		
		// initialize network arrays
		featureNodes = new ArrayList<Node>();
		hiddenNodes = new ArrayList<ArrayList<Node>>();
		outputNodes = new ArrayList<Node>();
		
		// add specified number of hidden layers to network
		for (int i = 0; i < hLayers; i++) {
			ArrayList oneHLayer = new ArrayList<Node>();
			hiddenNodes.add(oneHLayer);
		}
	}
	
	/**
	 * Set learning rate for classifier
	 * @param learning rate
	 */
	public void setEta(double learningRate) {
		this.n = learningRate;
	}
	
	/**
	 * Set number of iterations for classifier
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}
	
	/**
	 * Set bias offset value (below 1)
	 * @param offset value
	 */
	public void setBiasOffSet(double offset) {
		this.biasOffset = offset;
	}
	
	/**
	 * Set lambda
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	
	/**
	 * Set activation function to to specified type
	 */
	public void setActivation(int type) {
		if (type == 0) {
			this.reluActivation = true;
			this.LreluActivation = false;
			this.eluActivation = false;
			this.seluActivation = false;
		} else if (type == 1) {
			this.reluActivation = false;
			this.LreluActivation = true;
			this.eluActivation = false;
			this.seluActivation = false;
		} else if (type == 2) {
			this.reluActivation = false;
			this.LreluActivation = false;
			this.eluActivation = true;
			this.seluActivation = false;
		} else {
			this.reluActivation = false;
			this.LreluActivation = false;
			this.eluActivation = false;
			this.seluActivation = true;
		}
	}
	
	/**
	 * Train this classifier based on the data set
	 * @param data set
	 */
	public void train(DataSet data) {
		
		
		// Get data set information
		DataSet dataWBias = data.getCopyWithBias();
		dataSetFeatures = data.getAllFeatureIndices().size();
		featureMap = data.getFeatureMap();
		
		// numLabels issue fix
		Set<Double> labels = data.getLabels();
		if (!(labels.contains(0.0))) {
			labels.add(0.0);
		}
		//System.out.println("Training Labels:" + labels.toString());
		numLabels = labels.size();
		
		buildEmptyNetwork(); // initialized network nodes
		addRandomWeights(); // add random weights to all feature/hidden nodes
		
		for (int iter = 0; iter < iterations; iter++) {
			Collections.shuffle(dataWBias.getData());
			
			for (Example e: dataWBias.getData()) {

				// forward propagate through network, get outputs before activation
				double[] inputToOutputs = outputNoActivation(e);
				
				// back propagate and update weights of all nodes
				backpropagate(inputToOutputs, e);
			}
		}
	}
	
	/**
	 * Start of back propagation, updates last hidden layer and starts recursion
	 * @param inputs: inputs to output nodes from forward propagation
	 * @param e: example
	 */
	private void backpropagate(double[] inputs, Example e) {
		
		int intlabel = (int) e.getLabel();
		
		// perform softmax activation of the raw inputs
		double[] outputs = softmaxActivation(inputs);
		
		// Find cross-entropy loss
		double trueOutput = outputs[intlabel];
		double loss = -Math.log(trueOutput);
		
		// Find error at output layer
		double[] outputError = new double[numLabels];
		for (int i = 0; i < numLabels; i++) {
			
			if (i == intlabel) {
				outputError[i] = outputs[i] - 1.0;
			} else {
				outputError[i] = outputs[i];
			}
			
		}
		
		// find new weights for last layer of hidden layers (going to output nodes)
		double[][] newOut = new double[hNodes + 1][numLabels];
		ArrayList lastLayer = hiddenNodes.get(hLayers-1);
		
		// find error/new weights of each hidden node in this layer
		double[] layerError = new double[hNodes];
		for (int k = 0; k <= hNodes; k++) {
			
			double sumError = 0.0;
			Node currentNode = (Node) lastLayer.get(k);
				
			for (int w = 0; w < numLabels; w++) {
				
				// find new weights
				double gradient = outputError[w] * currentNode.nodeValue;
				double oldWeight = currentNode.getWeight(w);
				double r = lambda * oldWeight;
				newOut[k][w] = oldWeight - n *(gradient+r);
				
				// find error 
				if (k < hNodes) {
					sumError += outputError[w] * oldWeight;
				}
			}
			// store error of this hidden node
			if (k < hNodes) {
				layerError[k] = ddxActivation(sumError);
			}
		}
		
		// Initialize variables for recursive back propagation
		int lastLayerIndex = hLayers - 1;
		double [][] newInputWeights =new double[dataSetFeatures + 1][hNodes];
		Map<Integer, double[][]> newHWeight = new HashMap<Integer, double[][]>();
		
		recursiveBackProp(lastLayerIndex, layerError, newHWeight, newInputWeights); // back propagate and find map of new weights
	
		// update weights in feature nodes
		for (int k = 0; k <= dataSetFeatures; k++) {
			Node currentNode = featureNodes.get(k);
			for (int w = 0; w < hNodes; w++) {
				double weight = newInputWeights[k][w];
				currentNode.updateWeight(w, weight);
			}
		}
		
		//update weights in first n - 1 hidden layers
		for (int onelayer = 0; onelayer < hLayers - 1; onelayer++) {
			//System.out.println("Hidden Layer: " + onelayer);
			ArrayList currentLayer = hiddenNodes.get(onelayer);
			double[][] newweights = newHWeight.get(onelayer);
			
			for (int k = 0; k <= hNodes; k++) {
				Node currentNode = (Node) currentLayer.get(k);
				for (int w = 0; w < hNodes; w++) {
					double weight = newweights[k][w];
					currentNode.updateWeight(w, weight);
				}
			}
		}
		
		// update weights of last hidden layer
		for (int k = 0; k <= hNodes; k++) {
			Node currentNode = (Node) lastLayer.get(k);
			for (int w = 0; w < numLabels; w++) {
				double weight = newOut[k][w];
				currentNode.updateWeight(w, weight);
			}
		}
	}
	
	/**
	 * Recursive back propagation, finding weights for these nodes pointing to front layer
	 * @param layer: the current layer we are propagating through
	 * @param layerError: the recursive vector of errors (for each hidden node), which is used to find new weights
	 * @param newHWeight: Map of new weights (matrix) for each of the hidden layers
	 * @param: newInputWeights: matrix of new weights for the input layer
	 */
	private void recursiveBackProp(int layer, double[] layerError, Map newHWeight, double[][] newInputWeights) {
	
		ArrayList frontLayer = hiddenNodes.get(layer);
		ArrayList backLayer;
		
		if (layer == 0) { //updating input node weights (base case)

			// iterate through feature nodes
			for (int k = 0; k < dataSetFeatures; k++) {
				Node currentNode = featureNodes.get(k); 
				
				// calculate node error
				double sumError = 0.0;
				for (int w = 0; w < hNodes; w++) {
					sumError += layerError[w] * currentNode.getWeight(w);
				}
				
				double nodeError = ddxActivation(sumError);
				
				// find new weights
				for (int w = 0; w < hNodes; w++) {
					double gradient = nodeError * currentNode.nodeValue;
					double oldWeight = currentNode.getWeight(w);
					double r = lambda * oldWeight;
					newInputWeights[k][w] = oldWeight - n*(gradient+r);
				}
			}
			
				// calcuate new bias weights
				Node biasNode = (Node) featureNodes.get(dataSetFeatures);
				for (int w = 0; w < hNodes; w++) {
					double biasGradient = layerError[w];  
					double oldBiasWeight = biasNode.getWeight(w);
					double r = lambda * oldBiasWeight;
					double newBiasWeight = oldBiasWeight - n*biasOffset *(biasGradient+r);
					newInputWeights[dataSetFeatures][w] = newBiasWeight;
				}

		} else {
			
			
			backLayer = hiddenNodes.get(layer-1); // layer nodes whose weights are updated
			double[][] newWweights = new double[hNodes + 1][hNodes];
			double[] newLayerError = new double[hNodes];
			for(int k = 0; k < hNodes; k++) {
				
				Node backNode = (Node) backLayer.get(k);
				
				// find error in back node
				double sumError = 0.0;
				for (int w = 0; w < hNodes; w++) {
					sumError += layerError[w] * backNode.getWeight(w);
					
				}
				double nodeError;
				nodeError = ddxActivation(sumError);
				newLayerError[k] = nodeError;
				
				 
				// calculate and store new weights for back node
				for (int w = 0; w < hNodes; w++) {
					double gradient = nodeError * backNode.nodeValue;
					double oldWeight = backNode.getWeight(w);
					double r = lambda * oldWeight;
					newWweights[k][w] = oldWeight - n*(gradient+r);
				}
			}
			
			// calculate and store new weights for bias node
			Node biasNode = (Node) backLayer.get(hNodes);
			for (int w = 0; w < hNodes; w++) {
				double biasGradient = layerError[w];  // Error from the corresponding node
			    double oldBiasWeight = biasNode.getWeight(w);
			    double r = lambda * oldBiasWeight; // L2 regularization
			    double newBiasWeight = oldBiasWeight - n*biasOffset *(biasGradient+r);
			    newWweights[hNodes][w] = newBiasWeight;
			}
			
			// add new weights to map
			int newLayerNum = layer - 1;
			newHWeight.put(newLayerNum, newWweights);
	
			// recursive case: haven't reached the first hidden layer
			recursiveBackProp(newLayerNum, newLayerError, newHWeight, newInputWeights);
		}
		
	}
	
	
	/**
	 * Recursive back propagation, finding weights for these nodes pointing to front layer
	 * @param backlayer: the nodes whose weights sum into the front node
	 * @param layerError: the recursive vector of errors (for each hidden node), which is used to find new weights
	 * @param hidden layer: whether or not the back layer is the input layer
	 * @return: the sum of the weights from the back layer nodes into the front layer, no activation
	 */
	private double weightSum(ArrayList backLayer, int frontNode, boolean hiddenLayer) {
	
		double sum = 0;
		if(hiddenLayer) {
			
			for (int k = 0; k <= hNodes; k++) {
				Node backNode = (Node) backLayer.get(k);
				double weight = backNode.getWeight(frontNode);
				sum+= weight*backNode.nodeValue;
			}
			
		} else {
			
			for (int k = 0; k <= dataSetFeatures; k++) {
				Node backNode = (Node) featureNodes.get(k);
				double weight = backNode.getWeight(frontNode);
				sum+= weight*backNode.nodeValue;
			}
		}
		return sum;
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
		
		//System.out.println(e.toString());
		
		double[] inputsNoA = outputNoActivation(e);
		double[] outputs = softmaxActivation(inputsNoA);
		
		double runningOut = outputs[0];
		//System.out.println("Output Node 0: " + runningOut);
		double label = 0.0;
		
		for (int i = 1; i < numLabels; i++) {
			double output = outputs[i];
			//System.out.println("Output Node " + (i) + ": " + output);
			if (output > runningOut) {
				runningOut = output;
				label = (double) (i);
			}
		}
		//System.out.println("Predicted Label: " + label);
		return label;
	}
	
	/**
	 * Calculate confidence/out of an example
	 * in the current network
	 * 
	 * @param hidden nodes index
	 */
	public double confidence(Example example) {
		return 0.0;
	}
	
	
	/**
	 * The start of front propagation of an example, calls other function to forward propagate
	 * through the hidden layers
	 * @param e: example
	 * @return: The inputs to the output nodes, without activation. One for each label in the data set
	 */
	private double[] outputNoActivation(Example e) {
		
		// iterate through input nodes
		double[] inputsH1 = new double[hNodes];
		for (int feat = 0; feat <= dataSetFeatures; feat++) {
			double featureValue = e.getFeature(feat);
			Node inputNode = featureNodes.get(feat);
			if (feat != dataSetFeatures) {
				inputNode.setNodeValue(featureValue);
			} else {
				inputNode.setNodeValue(1);
			}
			
			//iterate through weight vectors 
			for (int w = 0; w < hNodes; w++) {
				inputsH1[w] += featureValue*inputNode.getWeight(w);
			}
			
		}
//		System.out.println("");
//		System.out.println("H0 Inputs");
//		for (int i = 0; i < hNodes; i++) {
//	          System.out.print(inputsH1[i]);
//	          
//	          if (i < hNodes - 1) {
//	               System.out.print(" ");
//	          }
//	        }
//		System.out.println("");
		
		// forward propagate through hidden layers
		forwardinHNodes(inputsH1);
		
		// calculate inputs to output nodes
		double[] outputs = new double[numLabels];
		ArrayList lastLayer = hiddenNodes.get(hLayers -1);
		
		for (int k = 0; k <= hNodes; k++) {
			Node currentNode = (Node) lastLayer.get(k);
			
			// iterate through weights
			for (int w = 0; w < numLabels; w++) {
				double weight = currentNode.getWeight(w);
				outputs[w] += currentNode.nodeValue * weight;
			}
		}

		return outputs;
	}
	
	// sets values of last hidden nodes
	/**
	 * Recursive back propagation, finding weights for these nodes pointing to front layer
	 * @param inputs: The inputs from the input layer, without activation, into the first hidden layer
	 */
	private void forwardinHNodes(double[] inputs) {
		ArrayList firstLayer = hiddenNodes.get(0);
		
		//apply activation function on inputs to first hidden layer
		//System.out.println("H0 Node Values:");
		for (int k = 0; k < hNodes; k++) {
			double act = activation(inputs[k]);
			Node currentNode = (Node) firstLayer.get(k);
			currentNode.setNodeValue(act);
//			System.out.print(	"Node " + k + ": " + currentNode.nodeValue);
//			System.out.print(" ");
		}
		//System.out.println("");
		// forward propagate through hidden layers
		for (int layer = 0; layer < hLayers - 1; layer++) {
			forwardOneLayer(layer);
		}
	}
	
	/**
	 * Move through hidden layers, setting the node values
	 */
	private void forwardOneLayer(int hLayer) {
		
		ArrayList currentLayer = hiddenNodes.get(hLayer);
		double[] inputsNext = new double[hNodes];
		
		// iterate through this layer's nodes to find inputs to next layer's nodes
		for (int k = 0; k <= hNodes; k++) {
			Node currentNode = (Node) currentLayer.get(k);
			
			//iterate through weight vectors
			for (int w = 0; w < hNodes; w++) {
				double weight = currentNode.getWeight(w);
				inputsNext[w] += currentNode.nodeValue*weight;
			}
		}
//		System.out.println("H" + (hLayer+1) + " Inputs:");
//		for (int i = 0; i < hNodes; i++) {
//	          System.out.print(inputsHk[i]);
//	          
//	          if (i < hNodes - 1) {
//	               System.out.print(" ");
//	          }
//	        }
//		System.out.println("");

		// apply activation on all inputs of the next layer
		//System.out.println("H" + (hLayer+1) + " Node Value:");
		ArrayList nextLayer = hiddenNodes.get(hLayer + 1);
		for (int k = 0; k < hNodes; k++) {
			double act = activation(inputsNext[k]);
			// set values of next layer's nodes
			Node currentNode = (Node) nextLayer.get(k);
			currentNode.setNodeValue(act);
//			System.out.print(	"Node " + k + ": " + currentNode.nodeValue);
//			System.out.print(" ");
		}

	}
	
	/**
	 * Add empty nodes to every layer of the network
	 * Clears network if already occupied
	 */
	private void buildEmptyNetwork() {
		
		// clear node arrays
		if (!featureNodes.isEmpty()) { featureNodes.clear();}
		if (!outputNodes.isEmpty()) { hiddenNodes.clear();}
		for (int i = 0; i < hLayers; i++) {
			ArrayList<Node> layer = hiddenNodes.get(i);
			if (!layer.isEmpty()) {layer.clear();}
		}
		
		// Populate input node list
		for (int i = 0; i < dataSetFeatures; i++) {
			Node inputnode = new Node(i);
			featureNodes.add(inputnode);
		}
		// Add bias node to input node list
		int biasIndex = dataSetFeatures;
		Node biasnode = new Node(biasIndex);
		biasnode.setNodeValue(1.0);
		featureNodes.add(biasnode);
		
		
		for (int l = 0; l < hLayers; l++) {
			ArrayList<Node> oneHLayer = hiddenNodes.get(l);
			
			// Populate hidden layer
			for (int i = 0; i < hNodes; i++) {
				Node hiddennode = new Node(i);
				oneHLayer.add(hiddennode);
			}
			// Add bias node to hidden node list
			int hBiasIndex = hNodes;
			Node hBiasnode = new Node(hBiasIndex);
			hBiasnode.setNodeValue(1.0);
			oneHLayer.add(hBiasnode);
		}
		
		// Populate output node list
		for (int i = 0; i < numLabels; i++) {
			Node outputNode = new Node(i);
			outputNodes.add(outputNode);
		}

	}
	
	/**
	 * Add random weights to all nodes in the vector
	 * Uses Kaiming initialization 
	 */
	private void addRandomWeights() {
		Random randomGenerator = new Random();
		
		// weights for featureNodes
		for (int j = 0; j <= dataSetFeatures; j++) {
			Node currentNode = featureNodes.get(j);
			for (int k = 0; k < hNodes; k++) {
				double variance = 2.0 / (double) (dataSetFeatures + 1);
				double strd = Math.sqrt(variance);
				double randNum = strd * randomGenerator.nextGaussian();
				currentNode.addWeight(randNum);
			}
		}
		
		// weights for hidden layers connected to hidden layers
		for (int i = 0; i < hLayers - 1; i++) {
			//System.out.println("Hidden Layer: " + (i+1));
			ArrayList layer = hiddenNodes.get(i);
			
			for (int k = 0; k <= hNodes; k++) {
				
				Node currentNode = (Node) layer.get(k);
				
				for (int w = 0; w < hNodes; w++) {
					double variance = 2.0 / (hNodes + 1);
	                double strd = Math.sqrt(variance);
	                double randNum = strd * randomGenerator.nextGaussian();
					currentNode.addWeight(randNum);
				}
			}
		}
		
		// weights for last hidden layer
		ArrayList lastLayer = hiddenNodes.get(hLayers - 1);
		for (int k = 0; k <= hNodes; k++) {
			Node currentNode = (Node) lastLayer.get(k);
			
			for (int o = 0; o < numLabels; o++) {
				double variance = 2.0 / (hNodes + 1); 
	            double strd = Math.sqrt(variance);
	            double randNum = strd * randomGenerator.nextGaussian();
				currentNode.addWeight(randNum);
			}
			
		}
	}
	
	/**
	 *  Activation function calculations
	 *  @return: the activation of the input double
	 */
	private double activation(double x) {
		
		// constants for ELU and SELU activation
		
	    if (reluActivation) {
	        return Math.max(0, x); // ReLU
	    } else if (LreluActivation) {
	        return x > 0 ? x : 0.01 * x; // Leaky ReLU
	    } else if (eluActivation) {
	        return x > 0 ? x : ALPHA * (Math.exp(x) - 1); // ELU
	    } else { // SELU
	        return x > 0 ? LAMBDA * x : LAMBDA * SELU_ALPHA * (Math.exp(x) - 1); // SELU
	    }
	}
	
	/**
	 *  Derivative of activation function 
	 *  @return: the derivative activation of the input double
	 */
	private double ddxActivation(double outputSum) {
		 
		 if (reluActivation) {
		        return outputSum > 0 ? 1 : 0; // ReLU
		    } else if (LreluActivation) {
		        return outputSum > 0 ? 1 : 0.01; // Leaky ReLU
		    } else if (eluActivation) {
		        return outputSum > 0 ? 1 : ALPHA * Math.exp(outputSum); // ELU
		    } else { // SELU
		        return outputSum > 0 ? LAMBDA : LAMBDA * SELU_ALPHA * Math.exp(outputSum); // SELU
		    }
	}
	
	
	/**
	 *  Softmax activation for output layer
	 *  @param inputs: an array used in softmax calculation
	 *  @return: an array (of the same size as the input array) of activated doubles
	 */
	private double[] softmaxActivation(double[] inputs1) {
	    double sum = 0.0;
	    double[] vector = new double[inputs1.length];

	    for (int i = 0; i < inputs1.length; i++) {
	        double output = Math.exp(inputs1[i]);
	        sum += output;
	        vector[i] = output;
	    }
	    
	    for (int i = 0; i < inputs1.length; i++) {
	        vector[i] /= sum;
	    }
	    
	    return vector;
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
		System.out.println("Feature Bias: " + featureNodes.get(dataSetFeatures).nodeValue);
		for (int k = 0; k < hNodes; k++) {
			
			System.out.println("Bias Weight " + (k+1) +": " + featureNodes.get(dataSetFeatures).getWeight(k));
		}
		System.out.println("");
		for (int l = 0; l < hLayers - 1; l++) {
			System.out.println("HIDDEN LAYER " + (l+1) + " –––");
			ArrayList layer = hiddenNodes.get(l);
			
			for (int k = 0; k < hNodes; k++) {
				Node currentNode = (Node) layer.get(k);
				System.out.println("    Hidden Node " + (k+1));
				
				for (int w = 0; w < hNodes; w++)
				System.out.println("	    Weight "+ (w+1) + ": " + currentNode.getWeight(w));
			}
			
			Node biasNode = (Node) layer.get(hNodes);
			System.out.println("    Bias Node: " + biasNode.nodeValue);
			for (int w = 0; w < hNodes; w++) {
				//System.out.println("	    Bias Weight "+ (w+1) + ": " + biasNode.getWeight(w));
			}
		}
		System.out.println("HIDDEN LAYER " + (hLayers) + " –––");
		ArrayList lastLayer = hiddenNodes.get(hLayers - 1);
		for (int k = 0; k < hNodes; k++) {
			Node currentNode = (Node) lastLayer.get(k);
			System.out.println("    Hidden Node " + (k+1));
			for (int w = 0; w < numLabels; w++) {
				//System.out.println("	    Weight "+ (w+1) + ": " + currentNode.getWeight(w)); 
			}
		}
		
		
		Node lastBias = (Node) lastLayer.get(hNodes);
		System.out.println("    Bias Node: " + lastBias.nodeValue);
		for (int w = 0; w < numLabels; w++) {
			System.out.println("	    Bias Weight "+ (w+1) + ": " + lastBias.getWeight(w));
		}
		
		
	}
	
	/**
	 * A class representing a node with an internal value
	 * weights directed out of it
	 */
	private static class Node {
		public double nodeValue;
		
		//public boolean hiddenNode;
		public ArrayList<Double> vectors;
		
		
		/**
		 *  Node Constructor
		 *  
		 *  @param node index
		 *  @param hiddenNode: if this node a hidden node
		 */
		public Node(int nodeNum) {
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
				vectors.set(toHNode, weight);

		}
		
		/**
		 *  Retrieve weight vector going to specified hidden node
		 *  or output node
		 */
		public double getWeight(int toHNode) {
			return vectors.get(toHNode);
		}
	
	}
}
