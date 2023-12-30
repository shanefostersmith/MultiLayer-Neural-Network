// Shane Foster-Smith, CS158, Assignment 8

import ml.data.*;
import ml.classifiers.*;


/**
 * Experiment class
 * Performs various experiments and outputs the results
 * ,
 */

public class Experiment {
	
	// Constructor
	public Experiment() {
		
	}
	
	/**
	 * Retrieve training and testing accuracy for 5-fold set
	 */
	public void changingNodeSize(CrossValidationSet splitset, int hLayers, int hNodes, int activation, double learningRate, int iterations, double lambda, double offset) {
		//System.out.println("inputL = " + hLayers);
		//System.out.println("inputN = " + hNodes);
		
		double totalAcc = 0.0;
		for (int splitNum = 0; splitNum < 5; splitNum++) {
			
			DataSetSplit split = splitset.getValidationSet(splitNum);
			
			// setup specified network
			MultiLayerNN mNN = new MultiLayerNN(hLayers, hNodes);
			mNN.setActivation(activation);
			mNN.setEta(learningRate);
			mNN.setIterations(iterations);
			mNN.setLambda(lambda);
			mNN.setBiasOffSet(offset);
			
			
			// Pre-Process data
			FeatureNormalizer fNormalizer = new FeatureNormalizer();
			ExampleNormalizer eNormalizer = new ExampleNormalizer();
			fNormalizer.preprocessTrain(split.getTrain());
			eNormalizer.preprocessTrain(split.getTrain());
			fNormalizer.preprocessTest(split.getTest());
			eNormalizer.preprocessTest(split.getTest());
			
			// train model
			mNN.train(split.getTrain());
			
			int spCorrect = 0;
			int spTotal = 0;
			for (Example e: split.getTest().getData()) {
				spTotal++;
				double prediction = mNN.classify(e);
				double label = e.getLabel();
				
				if (spTotal <= 10) {
					//System.out.println("prediction = " + prediction + " || label: " + label);
				}
//				
				if (prediction == label) {
					spCorrect++;
				}
			}
			
			double splitAcc = (double) spCorrect / (double) spTotal;
			System.out.println("Split " + splitNum + ": " + splitAcc);
			totalAcc += splitAcc;
			mNN.networkToString();
			
		}
		
		totalAcc = totalAcc*(0.2);
		System.out.print("Layers =  " + hLayers + " || ");
		System.out.print("Nodes =  " + hNodes);
		System.out.println("");
		System.out.println("Average Accuracy of Folds: " + totalAcc);
		System.out.println("");
		
		
	}
	
	public void changingNodeSize2(DataSetSplit splitset, int hLayers, int hNodes, int activation, double learningRate, int iterations, double lambda, double offset) { 
		// setup specified network
		MultiLayerNN mNN = new MultiLayerNN(hLayers, hNodes);
		mNN.setActivation(activation);
		mNN.setEta(learningRate);
		mNN.setIterations(iterations);
		mNN.setLambda(lambda);
		mNN.setBiasOffSet(offset);
					
					
		// Pre-Process data
		FeatureNormalizer fNormalizer = new FeatureNormalizer();
		ExampleNormalizer eNormalizer = new ExampleNormalizer();
		fNormalizer.preprocessTrain(splitset.getTrain());
		eNormalizer.preprocessTrain(splitset.getTrain());
		fNormalizer.preprocessTest(splitset.getTest());
		eNormalizer.preprocessTest(splitset.getTest());	
		
		// train model
		mNN.train(splitset.getTrain());
		
		int spCorrect = 0;
		int spTotal = 0;
		for (Example e: splitset.getTest().getData()) {
			spTotal++;
			double prediction = mNN.classify(e);
			double label = e.getLabel();
			
			if (spTotal <= 10) {
				//System.out.println("prediction = " + prediction + " || label: " + label);
			}
			
			if (prediction == label) {
				spCorrect++;
			}
		}
		System.out.print("Layers =  " + hLayers + " || ");
		System.out.print("Nodes =  " + hNodes);
		System.out.println("");
		double splitAcc = (double) spCorrect / (double) spTotal;
		System.out.println("Accuracy = " + splitAcc);
		System.out.println("");
		mNN.networkToString();
	}
	

}