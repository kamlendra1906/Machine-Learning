	/**
 * 
 */
package com.ml.hw5.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.ml.hw5.data.DataForRegression;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.data.Data;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class ClassifierUtil {
	
	public static final String SPAMBASE_POLLUTED_TRAINING_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\train_feature.txt";
	public static final String SPAMBASE_POLLUTED_TRAINING_LABEL_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\train_label.txt";
	public static final String SPAMBASE_POLLUTED_TEST_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\test_feature.txt";
	public static final String SPAMBASE_POLLUTED_TEST_LABEL_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\test_label.txt";
	public static final String SPAMBASE_ORIGINAL_TRAINING_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spambase\\spambase.data";
	public static final String SPAMBASE_ORIGINAL_FEATURE_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spambase\\spambase.names";
	public static final String SPAMBASE_POLLUTED_TRAINING_PCA = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\trainPCA.txt";
	public static final String SPAMBASE_POLLUTED_TEST_PCA = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\spam_polluted\\testPCA.txt";
	public static final String SPAMBASE_MISSING_TRAINING_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Spambase_missing_value\\Training\\20_percent_missing_train.txt";
	public static final String SPAMBASE_MISSING_TRAINING_FEATURE_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Spambase_missing_value\\Training\\spambase.names";
	public static final String SPAMBASE_MISSING_TEST_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Spambase_missing_value\\Test\\20_percent_missing_test.txt";
	public static final String DIGIT_TRAINING_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Digit\\Training\\train-images.idx3-ubyte";
	public static final String DIGIT_TRAINING_LABEL_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Digit\\Training\\train-labels.idx1-ubyte";
	public static final String DIGIT_TEST_DATA_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Digit\\Test\\t10k-images.idx3-ubyte";
	public static final String DIGIT_TEST_LABEL_FILE = 
			"C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework5\\Data\\Digit\\Test\\t10k-labels.idx1-ubyte";
	
		
	
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
	
	public static double testWeight(final DataForRegression testData, final Matrix weight, boolean useMSE) throws Exception {
		Matrix x = new Matrix(testData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(testData.getValueData(), testData.getSampleSize());
		Matrix predictedY = x.times(weight);
		if(useMSE) {
			return calculateMSE(y, predictedY);
		}
		return calculateAccuracy(y, predictedY);	
	}
	
	private static double calculateMSE(Matrix y, Matrix predictedY) {
		Matrix errorMatrix = predictedY.minus(y);
		int rows = errorMatrix.getRowDimension();
		int columns = errorMatrix.getColumnDimension();
		double squaredError = 0;
		for(int i=0; i<rows;i++) {
			for(int j=0;j<columns; j++) {
				squaredError+= Math.pow(errorMatrix.getArray()[i][j], 2);
			}
		}
		double returnValue = squaredError/rows;
		return returnValue;
	}
	
	private static double calculateAccuracy(Matrix y, Matrix predictedY) {
		Matrix predictedClass = new Matrix(predictedY.getRowDimension(), predictedY.getColumnDimension());
		double threshold = 0.5;
		for(int row=0; row < predictedY.getRowDimension(); row++) {
			for(int col=0; col < predictedY.getColumnDimension(); col++) {
				predictedClass.set(row, col, predictedY.get(row, col) > threshold ? 1 : 0);
			}
		}
		double error = 0;
		for(int row=0; row < predictedY.getRowDimension(); row++) {
			for(int col=0; col < predictedY.getColumnDimension(); col++) {
				error+= (y.get(row, col) != predictedClass.get(row, col)) ? 1 : 0;
			}
		}
		double errorPercentage = error/y.getRowDimension()*100;
		return 100 - errorPercentage;
	}
	
	public static DataForRegression prepareData(DataSet dataSet) throws Exception {
		if (dataSet != null && dataSet.dataSize() > 0) {
			List<Data> data = dataSet.getData();
			DataForRegression dataForRegression = new DataForRegression();
			double[] featureMatrix = null;
			double[] valueMatrix = null;
			int sampleSize = data.size();
			int featureSize = dataSet.getFeatures().size();
			
			int size = sampleSize * featureSize;
			featureMatrix = new double[size];
			double[][] twoDFeatureMatrix = new double[sampleSize][featureSize];
			valueMatrix = new double[data.size()];
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < featureSize; j++) {
					if(j ==0) {
						featureMatrix[featureSize * i + j] = 1;
						twoDFeatureMatrix[i][j] = 1;
					} else {
						featureMatrix[featureSize * i + j] = data.get(i).getFeatureValue(j-1);
						twoDFeatureMatrix[i][j] = data.get(i).getFeatureValue(j-1);
					}
				}
				valueMatrix[i] = data.get(i).labelValue();
			}
			dataForRegression.setFeatureData(featureMatrix);
			dataForRegression.setTwoDArrayFeatureData(twoDFeatureMatrix);
			dataForRegression.setValueData(valueMatrix);
			dataForRegression.setFeatureSize(featureSize);
			dataForRegression.setSampleSize(sampleSize);
			return dataForRegression;
		}
		return null;
	}
	
	public static void writeDataToFile(String fileLocation, DataSet dataSet) throws Exception {
		File file = new File(fileLocation);
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		for(Data data : dataSet.getData()) {
			bw.write(data.toString()+"\n");
		}
		bw.close();
	}
}