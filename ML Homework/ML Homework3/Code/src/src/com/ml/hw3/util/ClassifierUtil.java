	/**
 * 
 */
package src.com.ml.hw3.util;

import java.util.ArrayList;
import java.util.List;

/**
 * @author kkumar
 *
 */
public class ClassifierUtil {
	
	public static final String SPAM_FEATURES_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework3\\Data\\SpamBase\\spambase.names";
	public static final String SPAM_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework3\\Data\\SpamBase\\spambase.data";
	public static final String EM_TRAINING_FILE_2_FEATURE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework3\\Data\\Gaussian\\2gaussian.txt";
	public static final String EM_TRAINING_FILE_3_FEATURE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework3\\Data\\Gaussian\\3gaussian.txt";
	
	
	public static final int TRUE_POSITIVE = 0;
	public static final int FALSE_NEGATIVE = 1;
	public static final int FALSE_POSITIVE= 2;
	public static final int TRUE_NEGATIVE = 3;
	
	public static double logValue(double probability) {
		//return Math.log(probability)/Math.log(2);
		return Math.log(probability);
	}
	
		
	public static String printArray(double[] array) {
		StringBuilder builder = new StringBuilder();
		for(double d : array) {
			builder.append(d+"\t");
		}
		return builder.toString();
	}
	
	
	public static void updateConfusionMatrix(double[] confusionMatrix, double actualValue, double predictedValue) {
		if(actualValue == 1 && predictedValue == 1) {
			confusionMatrix[ClassifierUtil.TRUE_POSITIVE]+= 1;
		}
		if(actualValue == 1 && predictedValue == 0) {
			confusionMatrix[ClassifierUtil.FALSE_NEGATIVE]+= 1;
		}
		if(actualValue == 0 && predictedValue == 1) {
			confusionMatrix[ClassifierUtil.FALSE_POSITIVE]+= 1;
		}
		if(actualValue == 0 && predictedValue == 0) {
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
}