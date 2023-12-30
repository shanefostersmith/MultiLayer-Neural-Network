// Shane Foster-Smith, Assignment 4, CS158

package ml.data;

import ml.data.*;
import java.util.ArrayList;
import java.util.Set;
import ml.classifiers.*;

/**
 * A class that length normalizes the examples of a
 * training or testing data set
 */
public class ExampleNormalizer implements DataPreprocessor {
	
	private int totalFeatures;
	
	/**
	 * A function that normalizes the length of every example in a training data set
	 * @param: data set of examples
	 */
	public void preprocessTrain(DataSet data) {
		
		// Get number of features in data set and initialize dataList
		Set<Integer> feature_indices = data.getFeatureMap().keySet();
		totalFeatures = feature_indices.size();
		ArrayList<Example> dataList = (ArrayList<Example>) data.getData();
		
		calculateLength(dataList);
	}
	
	/**
	 * A function that normalizes the length of every example in a test data set
	 * @param: data set of examples
	 */
	public void preprocessTest(DataSet test) {
		
		// Get number of features in data set and initialize dataList
		ArrayList<Example> dataList = (ArrayList<Example>) test.getData();
		calculateLength(dataList);
	}
	
	/**
	 * A function that calculates the length of an example and
	 * divides each feature value in that example by the length
	 * @param: dataList: a list of examples
	 */
	private void calculateLength(ArrayList<Example> dataList) {
		
		for (Example example : dataList) {
					
			//calculate length
			double sumDistance = 0;
			for (int i = 0; i < totalFeatures; i++) {
				sumDistance += (example.getFeature(i))*(example.getFeature(i));
			}
			
			double length = Math.sqrt(sumDistance);
					
			// Iterate through each feature and divide value by length
			for (int j = 0; j < totalFeatures; j++) {
						
				double newValue = example.getFeature(j) / length;
				example.setFeature(j, newValue);
	
			}
		}
	}
}