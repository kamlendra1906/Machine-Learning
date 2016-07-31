package com.ml.hw1.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import com.ml.hw1.feature.Feature;
import com.ml.hw1.feature.FeatureReader;

/**
 * @author Kamlendra Kumar
 *
 */
public class DataInput {

	private static double[] featureValueMax;
	private static double[] featureValueMin;

	public static DataSet getDataFromFile(String dataFile, String featureFile, boolean normalizeData, boolean isTrainingData) throws Exception {
		BufferedReader readFile = null;
		DataSet dataset = null;
		List<Feature> features = new ArrayList<Feature>();
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			features = FeatureReader.getFeaturesList(featureFile);
			dataset = new DataSet(features.size() - 1, features);

			featureValueMax = new double[features.size()];
			featureValueMin = new double[features.size()];
			for(int i=0;i<features.size(); i++) {
				featureValueMax[i] = Double.NEGATIVE_INFINITY;
				featureValueMin[i] = Double.POSITIVE_INFINITY;
			}
			
			while (true) {
				String line = readFile.readLine();
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);

				double[] featureValues = new double[values.length];
								
				for (int i = 0; i < values.length; i++) {
					featureValues[i] = Double.parseDouble(values[i]);
					if(featureValues[i] < featureValueMin[i]) {
						featureValueMin[i] = featureValues[i];
					}
					if(featureValues[i] > featureValueMax[i]) {
						featureValueMax[i] = featureValues[i];
					}
				}

				Data data = new Data(featureValues);
				dataset.addData(data);

			}
		} finally {
			readFile.close();
		}
		return normalizeData ? shiftAndScaleNormalize(dataset) : dataset;
	}
	
	private static DataSet shiftAndScaleNormalize(DataSet dataSet) {
		for(Data data : dataSet.getData()) {
			double[] featureValues = data.getFeatureValues();
			for(int feature=0; feature< featureValues.length; feature++) {
				if(feature != dataSet.getLabelIndex()) {
					featureValues[feature] = (featureValues[feature] - featureValueMin[feature])/(featureValueMax[feature] - featureValueMin[feature]);
				}
			}
		}
		return dataSet;
	}
}