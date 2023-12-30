// Shane Foster-Smith, Assignment 4, CS158

package ml.data;

import ml.data.*;
import java.util.ArrayList;
import java.util.Set;
import ml.classifiers.*;

/**
 * A class that does feature re-scaling to a training and test data
 */
public class FeatureNormalizer implements DataPreprocessor {
	
	private int totalFeatures;
	double[] meanValues;
	double[] stdValues;
	
	/**
	 * A function that calculates and stored the mean and standard deviation values
	 * for each feature. Then, feature re-scaling is done on for each example.
	 * @param: data set of examples
	 */
	public void preprocessTrain(DataSet data) {
		
		// Get number of features in data set and initialize dataList
		Set<Integer> feature_indices = data.getFeatureMap().keySet();
		totalFeatures = feature_indices.size();
		ArrayList<Example> dataList = (ArrayList<Example>) data.getData();
		
		// Initialize variables to store mean and std values
		meanValues = new double[totalFeatures];
		stdValues = new double[totalFeatures];
		int numExamples = 0;
		boolean first_feature = true;
		
		// Calculate mean and variance for each feature and re-scale examples
		for (int i = 0; i < totalFeatures; i++) {
			
			// Calculate mean of feature
			double featureMean = 0;
			for (Example example : dataList) {
				
				featureMean += example.getFeature(i);
				if (first_feature) {
					numExamples++;
				}
			}
			if(first_feature) {
				first_feature = false;
			}
			
			featureMean = featureMean / (double) numExamples;
			meanValues[i] = featureMean;
			
			// Calculate standard deviation of feature and re-center
			double featureStd = 0;
			for (Example example : dataList) {
				
				featureStd += (example.getFeature(i) - featureMean)*(example.getFeature(i) - featureMean);
				
				// re-center feature
				double newFeatureValue = example.getFeature(i) - featureMean;
				example.setFeature(i, newFeatureValue);
				
			}
			featureStd = Math.sqrt(featureStd / (double)(numExamples - 1));
			stdValues[i] = featureStd;
			
			// Variance Scaling
			for (Example example : dataList) {
				double newFeatureValue = example.getFeature(i) / featureStd;
				example.setFeature(i, newFeatureValue);
			}
		
		}
		
	}
	
	/**
	 * A function that does feature re-scaling for each example of data set.
	 * Uses the mean and standard deviation values calculated for the training data
	 * @param: data set of examples
	 */
	public void preprocessTest(DataSet test) {
		
		// Initialize dataList
		ArrayList<Example> dataList = (ArrayList<Example>) test.getData();
		
		// Re-center and re-scale each feature value
		for (int i = 0; i < totalFeatures; i++) {
			for (Example example : dataList) {
				
				// calculate new feature value and set this value
				double newFeatureValue = example.getFeature(i) - meanValues[i];
				newFeatureValue = newFeatureValue / stdValues[i];
				example.setFeature(i, newFeatureValue);
				
				
			}
		}
		
		
		
	}
}