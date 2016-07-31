package com.ml.hw4.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import com.ml.hw4.feature.Feature;
import com.ml.hw4.feature.FeatureReader;

import Jama.Matrix;

/**
 * @author Kamlendra Kumar
 */
public class DataInput {

	public static DataSet getData(String dataFile, String featureFile) throws Exception {
		BufferedReader readFile = null;
		DataSet dataset = null;
		List<Feature> features = new ArrayList<Feature>();
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			features = FeatureReader.getFeaturesList(featureFile);
			dataset = new DataSet(features.size() - 1, features);

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
				}
				
				Data data = new Data(featureValues);
				dataset.addData(data);

			}
		} finally {
			readFile.close();
		}
		return dataset;
	}
	
	public static DataSet getDataForECOC(String dataFile) throws Exception {
		BufferedReader readFile = null;
		DataSet dataset = null;
		List<Feature> features = getFeaturesForECOC();
		
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			dataset = new DataSet(features.size() - 1, features);

			while (true) {
				String line = readFile.readLine();
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				double[] featureValues = new double[features.size()];

				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);

				featureValues[features.size() - 1] = Double.parseDouble(values[0]);
				
				for(int i=1; i < values.length; i++) {
					String[] array = values[i].split(":");
					int featureIndex = Integer.parseInt(array[0]);
					double featureValue = Double.parseDouble(array[1]);
					featureValues[featureIndex] = featureValue;
				}
				Data data = new Data(featureValues);
				dataset.addData(data);
			}
		} finally {
			readFile.close();
		}
		return dataset;
	}
	
	private static List<Feature> getFeaturesForECOC() {
		int totalFeatures = 1754;
		List<Feature> features = new ArrayList<Feature>();
		for(int i=0; i < totalFeatures; i++) {
			features.add(new Feature("i", Feature.NUMERICAL));
		}
		features.add(new Feature("label", Feature.NOMINAL));
		return features;
	}

	public static List<Matrix> getData(String dataFile) throws Exception {
		BufferedReader readFile = null;
		List<Matrix> data = new ArrayList<Matrix>();
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			
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
				}
				data.add(new Matrix(featureValues, featureValues.length));
			}
		} finally {
			readFile.close();
		}
		return data;
	}

	public static void normalizeData(DataSet trainingData, DataSet testData) throws Exception {
		double[] featureValueMax = null;
		double[] featureValueMin = null;
		if (trainingData != null && trainingData.dataSize() > 0) {
			int featureSize = trainingData.getFeatures().size();
			featureValueMax = new double[featureSize];
			featureValueMin = new double[featureSize];
			for (int i = 0; i < featureSize; i++) {
				featureValueMax[i] = Double.NEGATIVE_INFINITY;
				featureValueMin[i] = Double.POSITIVE_INFINITY;
			}

		}
		updateFeatureMinMaxValues(featureValueMax, featureValueMin, trainingData);
		updateFeatureMinMaxValues(featureValueMax, featureValueMin, testData);
		shiftAndScaleNormalize(trainingData, featureValueMin, featureValueMax);
		shiftAndScaleNormalize(testData, featureValueMin, featureValueMax);
	}

	private static void updateFeatureMinMaxValues(double[] featureValueMax, double[] featureValueMin, DataSet dataSet)
			throws Exception {
		if (dataSet != null && dataSet.dataSize() > 0) {
			for (Data data : dataSet.getData()) {
				for (int featureIndex = 0; featureIndex < dataSet.getFeatures().size(); featureIndex++) {
					double value = data.getFeatureValue(featureIndex);
					if (value > featureValueMax[featureIndex]) {
						featureValueMax[featureIndex] = value;
					}
					if (value < featureValueMin[featureIndex]) {
						featureValueMin[featureIndex] = value;
					}
				}
			}
		}
	}

	private static DataSet shiftAndScaleNormalize(DataSet dataSet, double[] featureValueMin, double[] featureValueMax)
			throws Exception {
		if (dataSet != null && dataSet.dataSize() > 0) {
			for (Data data : dataSet.getData()) {
				double[] featureValues = data.getFeatureValues();
				for (int feature = 0; feature < featureValues.length; feature++) {
					if (feature != dataSet.getLabelIndex()) {
						featureValues[feature] = (featureValues[feature] - featureValueMin[feature])
								/ (featureValueMax[feature] - featureValueMin[feature]);
					}
				}
			}
		}
		return dataSet;
	}
	
	/*private static List<Feature> getFeaturesForUCI(String dataFile, String featureFile) {
		
	}*/
	
}