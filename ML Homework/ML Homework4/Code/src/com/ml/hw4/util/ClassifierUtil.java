	/**
 * 
 */
package com.ml.hw4.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.ml.hw4.data.Data;

/**
 * @author kkumar
 *
 */
public class ClassifierUtil {
	
	public static final String SPAM_FEATURES_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\SpamBase\\spambase.names";
	public static final String SPAM_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\SpamBase\\spambase.data";
	public static final String HOUSING_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\Housing\\housing_train.txt";
	public static final String HOUSING_FEATURE_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\Housing\\housing_features.txt";
	public static final String HOUSING_TEST_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\Housing\\housing_test.txt";
	public static final String ECOC_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\8newsgroup\\train.trec\\feature_matrix.txt";
	public static final String ECOC_TEST_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\8newsgroup\\test.trec\\feature_matrix.txt";
	public static final String ECOC_TRAINING_FILE1 = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\8newsgroup - Copy\\train.trec\\feature_matrix.txt";
	public static final String ECOC_TEST_FILE1 = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework4\\Data\\8newsgroup - Copy\\test.trec\\feature_matrix.txt";
	
		
	
	public static final int TRUE_POSITIVE = 0;
	public static final int FALSE_NEGATIVE = 1;
	public static final int FALSE_POSITIVE= 2;
	public static final int TRUE_NEGATIVE = 3;
	
	public static double logValue(double probability) {
		return Math.log(probability)/Math.log(2);
		//return Math.log(probability);
	}
	
		
	public static String printArray(double[] array) {
		StringBuilder builder = new StringBuilder();
		for(double d : array) {
			builder.append(d+"\t");
			//builder.append(d+", " );
		}
		return builder.toString();
	}
	
	
	public static void updateConfusionMatrix(double[] confusionMatrix, double actualValue, double predictedValue) {
		if(actualValue == 1 && predictedValue == 1) {
			confusionMatrix[ClassifierUtil.TRUE_POSITIVE]+= 1;
		}
		if(actualValue == 1 && (predictedValue == 0 || predictedValue == -1)) {
			confusionMatrix[ClassifierUtil.FALSE_NEGATIVE]+= 1;
		}
		if((actualValue == 0 || actualValue == -1) && predictedValue == 1) {
			confusionMatrix[ClassifierUtil.FALSE_POSITIVE]+= 1;
		}
		if((actualValue == 0 || actualValue == -1) && (predictedValue == 0 || predictedValue == -1)) {
			confusionMatrix[ClassifierUtil.TRUE_NEGATIVE]+= 1;
		}
	}
	
	
	public static List<double[]> getROCCurveData(List<double[]> confusionMatrixData) {
		List<double[]> dataPoints = new ArrayList<double[]>();
		for(double[] confusionMatrix : confusionMatrixData) {
			dataPoints.add(getROCDataPoint(confusionMatrix));
		}
		return dataPoints;
	}

	public static double[] getROCDataPoint(double[] confusionMatrix) {
		double[] dataPoint = new double[2];
		double tp = confusionMatrix[ClassifierUtil.TRUE_POSITIVE];
		double fn = confusionMatrix[ClassifierUtil.FALSE_NEGATIVE];
		double fp = confusionMatrix[ClassifierUtil.FALSE_POSITIVE];
		double tn = confusionMatrix[ClassifierUtil.TRUE_NEGATIVE];
		
		double tpr = tp/(tp+fn);
		double fpr = fp/(fp+tn);
		dataPoint[0] = fpr;
		dataPoint[1] = tpr;
		return dataPoint;
	}

	public static void updateAverageConfusionMatrix(double[] averageConfusionMatrix, double[] confusionMatrix) {
		for(int i=0; i < 4; i++) {
			averageConfusionMatrix[i]+= confusionMatrix[i];
		}
	}
	
	public static void getAverage(double[] confusionMatrix, int fold) {
		for(int i=0; i< 4; i++) {
			confusionMatrix[i]/=fold;
		}
	}

	public static void normalizeProbability(double[] dataModelProbabilityArray, double totalProbability) {
		for(int i=0; i< dataModelProbabilityArray.length; i++) {
			dataModelProbabilityArray[i]/= totalProbability;
		}
	}
	
	public static List<Data> randomSampleWithoutReplacement(List<Data> items, double percent){
		Random random = new Random();
		int sampleSize = (int) (items.size() * percent/100);
		List<Data> result = new ArrayList<Data>();
		
		while(result.size() < sampleSize && items.size() > 0) {
			int randomLocation = random.nextInt(items.size() - 1);
			result.add(items.get(randomLocation));
			items.remove(randomLocation);
		}
		return result;
	}
	
	public static List<Data> randomSampleWithReplacement(List<Data> items, double percent){
		Random random = new Random();
		int sampleSize = items.size();
		List<Data> result = new ArrayList<Data>();
		
		while(result.size() < sampleSize) {
			int randomLocation = random.nextInt(items.size() - 1);
			result.add(items.get(randomLocation));
		}
		return result;
	}
}