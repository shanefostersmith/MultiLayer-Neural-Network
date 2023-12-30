// Shane Foster-Smith, CS158, Assignment 8

import ml.data.*;

import java.util.Collections;

import ml.classifiers.*;

/**
 * A main class that read in a data set from a file,
 * calls the experiments to be performed, and outputs the results
 * ,
 */

public class Main {
	
	public static void main(String[] args) { 
		String filename1 = "data/music1.csv";
		DataSet example1;
		try {
			example1 = new DataSet(filename1 , DataSet.CSVFILE);
		    System.out.println("Data loaded successfully");
		    } catch (Exception e) {
		    	System.err.println("Failed to load the dataset from train-titanic.csv.");
		        e.printStackTrace();
		        return;
		    }
		
		
		// Get training split
		
		
		// Bootstrap Resample
		BootstrapResample resampler = new BootstrapResample(example1, 300);
		DataSet newSet = resampler.newData();
		Collections.shuffle(newSet.getData());
		System.out.println("og labels " + newSet.getLabels().toString());
		
		DataSetSplit split1 = newSet.split(0.8);
		
		// Confirm new data set info
		int newNum = 0;
		int[] newNumEx = new int[newSet.getLabels().size()];
		for (Example e: newSet.getData()) {
			newNum++;
			int intlabel = (int) e.getLabel();
			newNumEx[intlabel]++;
		}
		
		System.out.println("NEW Training Examples = " + newNum);
		for (int p = 0; p < newSet.getLabels().size(); p++) {
			System.out.println("	Label " + p + ": " + newNumEx[p]);
		}
		
		// Conduct experiments
		CrossValidationSet validation = new CrossValidationSet(newSet, 5);
		Experiment experiment = new Experiment();
		
		// values to check
		int[] layers1 = {2};
		int[] nodes1 = {50};
		
		//constant iteration, learning rate and activation
		int activation = 3;
		double learningRate = 0.00001;
		int iterations = 250;
		double lambda = 0.001;
		double offset = .0001;
		
		for (Integer layer : layers1) {
			for (Integer node : nodes1) {
				experiment.changingNodeSize(validation, layer, node, activation, learningRate, iterations, lambda, offset);
			}
		}
		
		
		// constant node size and iterations, varying learning rate
//		double[] rates = {.0000001, .0000005, .000001, .000005, .00001, .00005, .0001};
//		int layers = 1;
//		int nodes = 5;
//		int activation = 3;
//		int iters = 50;
//		double lambda = 0.001;
//		
//		for (double n : rates) {
//			System.out.println("n = " + n);
//			experiment.changingNodeSize(validation, layers, nodes, activation, n, iters, lambda);
//		}
				
		
		
		
		//System.out.println("Labels: " + example1.getLabels());
//		MultiLayerNN mNN = new MultiLayerNN(2,10);
//		mNN.train(newSet);
//		mNN.networkToString();
//		
//		int correct = 0;
//		int total = 0;
//		int i = 0;
//		for (Example e: split1.getTest().getData()) {
//			double prediction = mNN.classify(e);
//			System.out.println(e.toString());
//			if (e.getLabel() == prediction) {
//				correct++;
//			}
//			//System.out.println("Actual Label: " + e.getLabel());
//			//System.out.println("Prediction: " + prediction);
//			total++;
//			
//			if (i > 5) {
//				break;
//			} else {
//				i++;
//			}
//			
//		}
//		
//		double accuracy = (double)correct / (double) total;
//		System.out.println("Accuracy: " + accuracy);
	}
}