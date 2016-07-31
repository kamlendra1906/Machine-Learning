package com.ml.hw6.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ml.hw6.data.Data;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.feature.Feature;
import com.ml.hw6.feature.FeatureReader;
import com.ml.hw6.feature.IFeature;

import Jama.Matrix;

/**
 * @author Kamlendra Kumar
 */
public class DataInput {

	private static int counter = 0;
	
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
					if("nan".equalsIgnoreCase(values[i])) {
						featureValues[i] = Double.NaN;
					} else {
						featureValues[i] = Double.parseDouble(values[i]);
					}
				}
				
				Data data = new Data(featureValues);
				data.setDataId(counter++);
				dataset.addData(data);

			}
		} finally {
			readFile.close();
		}
		return dataset;
	}
	
	public static DataSet getDigitData(String dataFile, int totalFeatures) throws Exception {
		BufferedReader readFile = null;
		DataSet dataset = null;
		List<Feature> features = new ArrayList<Feature>();
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			features = getFeatures(totalFeatures);
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
					if("nan".equalsIgnoreCase(values[i])) {
						featureValues[i] = Double.NaN;
					} else {
						featureValues[i] = Double.parseDouble(values[i]);
					}
				}
				Data data = new Data(featureValues);
				data.setDataId(counter++);
				dataset.addData(data);
			}
		} finally {
			readFile.close();
		}
		return dataset;
	}
	
	public static DataSet getDataForHW5(String dataFileName, String labelFileName) throws Exception {
		BufferedReader dataFile = null;
		BufferedReader labelFile = null;
		DataSet dataset = null;
		List<Feature> features = null;
		try {
			features = FeatureReader.getFeatureForHW5(dataFileName);
			dataset = new DataSet(features.size() - 1, features);

			dataFile = new BufferedReader(new FileReader(dataFileName));
			labelFile = new BufferedReader(new FileReader(labelFileName));
			
			while (true) {
				String dataLine = dataFile.readLine();
				String labelLine = labelFile.readLine();
				
				if (dataLine == null) {
					break;
				}

				if (dataLine.trim().length() == 0) {
					break;
				}
				String delims = "\\s+|,";
				String[] values = dataLine.trim().split(delims);

				double[] featureValues = new double[values.length+1];

				for (int i = 0; i < values.length; i++) {
					if("nan".equalsIgnoreCase(values[i])) {
						featureValues[i] = Double.NaN;
					} else {
						featureValues[i] = Double.parseDouble(values[i]);
					}
				}
				
				featureValues[featureValues.length - 1] = Double.parseDouble(labelLine);
				
				Data data = new Data(featureValues);
				dataset.addData(data);

			}
		} finally {
			dataFile.close();
			labelFile.close();
		}
		return dataset;
	}
	
	public static DataSet getDataForUCI(String dataFile, String featureFile) throws Exception {
		BufferedReader featureReader = null;
		BufferedReader dataReader = null;
		List<Feature> features = new ArrayList<Feature>();
		DataSet dataSet = null;
		Map<Integer, Map<String, Integer>> featureMapper = new HashMap<Integer, Map<String, Integer>>();
		Map<Integer, Integer> continuousFeatureMapper = new HashMap<Integer, Integer>();
		Map<String, Integer> labelIdMapper = new HashMap<String, Integer>();
		int[] featureTypeArray = null;
		try {
			
			// load features first 
			featureReader = new BufferedReader(new FileReader(featureFile));
			String line = featureReader.readLine();
			String delim = "\\s+|,";
			String[] counts = line.split(delim);
			
			int noOfDiscretefeature = Integer.parseInt(counts[1]);
			int noOfContinuosfeature = Integer.parseInt(counts[2]);
			int totalFeatures = noOfContinuosfeature + noOfDiscretefeature;
			featureTypeArray = new int[totalFeatures];
					
			
			int featureCounter = 0;
			for(int i = 0; i < totalFeatures; i++) {
				line = featureReader.readLine();
				String values[] = line.split(delim);
				
				if(Double.parseDouble(values[0]) < 0) {												// Continuous feature
					
					featureTypeArray[i] = IFeature.NUMERICAL;
					features.add(new Feature(String.valueOf(featureCounter), IFeature.NUMERICAL));
					continuousFeatureMapper.put(i, featureCounter++);
					
				} else {
					
					featureTypeArray[i] = IFeature.DISCRETE;										// Discrete feature
					features.add(new Feature(String.valueOf(featureCounter), IFeature.DISCRETE));
					
					int numberOfDiscreteValues = Integer.parseInt(values[0]);
					for(int value = 0; value < numberOfDiscreteValues; value++) {
						Map<String, Integer> discreteValueFeatureMap = featureMapper.get(i);
						if(discreteValueFeatureMap == null) {
							discreteValueFeatureMap = new HashMap<String, Integer>();
							featureMapper.put(i, discreteValueFeatureMap);
						} 
						discreteValueFeatureMap.put(values[value+1], value);
					}
					
				}
			}
			features.add(new Feature(String.valueOf(featureCounter), IFeature.LABEL));
			line = featureReader.readLine();
			
			String values[] = line.split(delim);
			int numberOfClasses = Integer.parseInt(values[0]);
			
			for(int i = 0; i < numberOfClasses; i++) {
				labelIdMapper.put(values[i+1], i);
			}
			
			// now loading data
			dataSet = new DataSet(features.size() -1 , features);
			dataReader = new BufferedReader(new FileReader(dataFile));
			List<Data> missingData = new ArrayList<Data>();
			
			while(true) {
				line = dataReader.readLine();
				if (line == null || line.trim().length() == 0) {
					break;
				}
				boolean hasMissingValues = line.indexOf("?") != -1;
				values = line.trim().split(delim);

				double[] featureValues = new double[features.size()];
				
				for(int i = 0; i < values.length -1; i++) {
					if(values[i].trim().equalsIgnoreCase("?")) {
						featureValues[i] = Double.NaN;
					} else {
						if(featureTypeArray[i] == IFeature.NUMERICAL) {
							featureValues[i] = Double.parseDouble(values[i]);
						} 
						if(featureTypeArray[i] == IFeature.DISCRETE) {
							featureValues[i] = featureMapper.get(i).get(values[i]);
						}
					}
				}
				
				featureValues[features.size()-1] = labelIdMapper.get(values[values.length - 1]);
				Data data = new Data(featureValues);
				
				if(hasMissingValues) {
					missingData.add(data);
				} else {
					dataSet.addData(data);
				}
			}
			
			// filling out missing values
			
			int featureSize = features.size() - 1;
			
			double[] featuresMean = dataSet.getDataFeatureMean();
			
			if(missingData != null && missingData.size() > 0) {
				
				for(Data data : missingData) {
					for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
						if(Double.isNaN(data.getFeatureValue(featureIndex))) {
							if(features.get(featureIndex).isDiscrete()) {
								data.setFeatureValue(featureIndex, Math.floor(featuresMean[featureIndex]));
							}
							if(features.get(featureIndex).isNumerical()) {
								data.setFeatureValue(featureIndex, featuresMean[featureIndex]);
							}
						}
					}
				}
				for(Data data : missingData) {
					dataSet.addData(data);
				}
			}
			
		} catch(Exception e) {
			throw e;
		} finally {
			featureReader.close();
			dataReader.close();
		}
		return dataSet;
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
	
	
	private static List<Feature> getFeatures(int totalFeatures) {
		List<Feature> features = new ArrayList<Feature>();
		for(int i=0; i < totalFeatures; i++) {
			features.add(new Feature("i", Feature.NUMERICAL));
		}
		features.add(new Feature("label", Feature.LABEL));
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

}