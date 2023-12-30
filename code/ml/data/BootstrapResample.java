package ml.data;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Bootstrap resampling class
 * Given a data set, provides a new one with balances labels
 * ,
 */
public class BootstrapResample {
	
	DataSet original;
	int numLabels;
	int examplePerLabel;
	
	public BootstrapResample(DataSet data, int examplePerLabel) {
		this.original = data;
		this.numLabels = data.getLabels().size();
		this.examplePerLabel = examplePerLabel;
	} 
	
	public DataSet newData() {
		
		DataSet newData = new DataSet(original.getFeatureMap());
		
		// find number of examples for each label
		int i = 0;
		int[] numEx = new int[numLabels];
		for (Example e: original.getData()) {
			
			i++;
			int intlabel = (int) e.getLabel();
			numEx[intlabel]++;
		}
		System.out.println("Num Training Examples = " + i);
		for (int p = 0; p < numLabels; p++) {
			System.out.println("	Label " + p + ": " + numEx[p]);
		}
		
		// get hashmap of datasets with just a single label
		Map dataMap = new HashMap<Integer, DataSet>();
		for (int p = 0; p < numLabels; p++) {
			DataSet data = new DataSet(original.getFeatureMap());
			dataMap.put(p, data);
		}
		
		// Fill datasets with examples
		for (Example e: original.getData()) {
				int intlabel = (int) e.getLabel();
				DataSet thisData = (DataSet) dataMap.get(intlabel);
				thisData.addData(e);
		}
		// Fill new data set with specified number of examples from each class
		for (int p = 0; p < numLabels; p++) {
			DataSet thisData = (DataSet) dataMap.get(p);
			
			if (numEx[p] >= examplePerLabel) { // if dataset already has enough of this label, choose them at random
				int ex = 0;
				Collections.shuffle(thisData.getData());
				for (Example e: thisData.getData()) {
					if (ex >= examplePerLabel) {
						break;
					} else {
						newData.addData(e);
						ex++;
					}
				}
			}
			else { // if there doesn't enough unique examples of this label
				
				// shuffle the data, pick first 5, and repeat 100 times to get 500 examples
				for (int k = 0; k < examplePerLabel; k++) {
					Collections.shuffle(thisData.getData());
					Example firstEx = thisData.getData().get(0);
					newData.addData(firstEx);
				}
			}
		}

		return newData;
		
	} // end function
		
		
}
	
