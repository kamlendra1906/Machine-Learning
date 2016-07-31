	/**
 * 
 */
package com.ml.hw2.util;

import java.util.List;

import com.ml.hw2.classifier.StochasticGradientDescentImpl;
import com.ml.hw2.data.Data;
import com.ml.hw2.data.DataForRegression;
import com.ml.hw2.data.DataSet;

import Jama.Matrix;

/**
 * @author kkumar
 *
 */
public class ClassifierUtil {
	
	public static final String HOUSING_FEATURES_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\Housing\\housing_features.txt";
	public static final String HOUSING_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\Housing\\housing_train.txt";
	public static final String HOUSING_TEST_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\Housing\\housing_test.txt";
	public static final String SPAM_FEATURES_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\SpamBase\\spambase.names";
	public static final String SPAM_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\SpamBase\\spambase.data";
	public static final String PERCEPTRON_TRAINING_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\Perceptron\\perceptronData.txt";
	public static final String PERCEPTRON_FEATURE_FILE = "C:\\Users\\kkumar\\Desktop\\ML Homework\\ML Homework2\\Data\\Perceptron\\perceptronFeature.txt";
	
	public static final int TRUE_POSITIVE = 0;
	public static final int FALSE_NEGATIVE = 1;
	public static final int FALSE_POSITIVE= 2;
	public static final int TRUE_NEGATIVE = 3;
	
	public static double logValue(double probability) {
		return Math.log(probability)/Math.log(2);
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
	
	public static double testWeight(final DataForRegression testData, final Matrix weight, boolean useMSE) throws Exception {
		Matrix x = new Matrix(testData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(testData.getValueData(), testData.getSampleSize());
		Matrix predictedY = x.times(weight);
		if(useMSE) {
			return calculateMSE(y, predictedY);
		}
		return calculateAccuracy(y, predictedY);	
	}
	
	public static Matrix getPredictedValueForLinearRegression(final DataForRegression testData, final Matrix weight) {
		Matrix x = new Matrix(testData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(testData.getValueData(), testData.getSampleSize());
		Matrix predictedY = x.times(weight);
		return predictedY;
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

	public static Matrix getPredictedClassValueForLogisticRegression(final DataForRegression testData, final Matrix weight, double threshold) {
		Matrix x = new Matrix(testData.getTwoDArrayFeatureData());
		Matrix y = new Matrix(testData.getValueData(), testData.getSampleSize());
		Matrix predictedY = x.times(weight);
		
		Matrix predictedClass = new Matrix(predictedY.getRowDimension(), predictedY.getColumnDimension());
		for(int row=0; row < predictedY.getRowDimension(); row++) {
			for(int col=0; col < predictedY.getColumnDimension(); col++) {
				predictedClass.set(row, col, getLogisticRegressionValue(predictedY.get(row, col)) > threshold ? 1 : 0);
			}
		}
		return predictedClass;
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
	
	public static String printArray(double[] array) {
		StringBuilder builder = new StringBuilder();
		for(double d : array) {
			builder.append(d+"\t");
		}
		return builder.toString();
	}
	
	public static double[] getConfusionMatrixForLinearRegression(int fold, DataSet trainingData, DataSet testData, double theta) throws Exception {
		double lambda = 0.001;
		double threshold = 0.00001;
		
		double[] confusionMatrixData = new double[4];
				
		StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , true);
		Matrix predictedValueMatrix = ClassifierUtil.getPredictedValueForLinearRegression(ClassifierUtil.prepareData(testData), weight);
		
		for(int dataIndex = 0; dataIndex < testData.getData().size(); dataIndex++) {
			Data data = testData.getData().get(dataIndex);
			double actualValue = data.labelValue();
			double predictedValue = predictedValueMatrix.get(dataIndex, 0);
			updateConfusionMatrix(confusionMatrixData, actualValue, predictedValue > theta ? 1 : 0);
		}
		return confusionMatrixData;
	}
	
	public static Matrix trainForLinearRegression(DataSet trainingData) throws Exception {
		double lambda = 0.001;
		double threshold = 0.00001;
		
		StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , true);
		return weight;
	}
	
	public static double[] getConfusionMatrixForLinearRegression(Matrix weight, DataSet testData, double theta)
			throws Exception {
		double[] confusionMatrixData = new double[4];
		Matrix predictedValueMatrix = ClassifierUtil.getPredictedValueForLinearRegression(ClassifierUtil.prepareData(testData), weight);

		for (int dataIndex = 0; dataIndex < testData.getData().size(); dataIndex++) {
			Data data = testData.getData().get(dataIndex);
			double actualValue = data.labelValue();
			double predictedValue = predictedValueMatrix.get(dataIndex, 0);
			updateConfusionMatrix(confusionMatrixData, actualValue, predictedValue > theta ? 1 : 0);
		}
		return confusionMatrixData;
	}
	
	public static double[] getConfusionMatrixForLogisticRegression(int fold, DataSet trainingData, DataSet testData, double theta) throws Exception {
		double lambda = 0.1;
		double threshold = 0.00001;
		
		double[] confusionMatrixData = new double[4];
		
		StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , false);
		
		Matrix predictedValueMatrix = ClassifierUtil.getPredictedClassValueForLogisticRegression(ClassifierUtil.prepareData(testData), weight, theta);
		
		for(int dataIndex = 0; dataIndex < testData.getData().size(); dataIndex++) {
			Data data = testData.getData().get(dataIndex);
			double actualValue = data.labelValue();
			double predictedValue = predictedValueMatrix.get(dataIndex, 0);
			updateConfusionMatrix(confusionMatrixData, actualValue, predictedValue);
		}
		return confusionMatrixData;
	}
	
	
	public static Matrix trainForLogisticRegression( DataSet trainingData) throws Exception {
		double lambda = 0.1;
		double threshold = 0.00001;
		StochasticGradientDescentImpl gradientDescent = new StochasticGradientDescentImpl(lambda, trainingData.getFeatures().size());
		Matrix weight = gradientDescent.findOptimalWeight(ClassifierUtil.prepareData(trainingData), threshold , false);
		return weight;
	}
	
	public static double[] getConfusionMatrixForLogisticRegression(Matrix weight, DataSet testData, double theta) throws Exception {
		double[] confusionMatrixData = new double[4];
		Matrix predictedValueMatrix = ClassifierUtil.getPredictedClassValueForLogisticRegression(ClassifierUtil.prepareData(testData), weight, theta);
		
		for(int dataIndex = 0; dataIndex < testData.getData().size(); dataIndex++) {
			Data data = testData.getData().get(dataIndex);
			double actualValue = data.labelValue();
			double predictedValue = predictedValueMatrix.get(dataIndex, 0);
			//System.out.println("theta= "+ theta+ " prediction="+ predictedValue);
			updateConfusionMatrix(confusionMatrixData, actualValue, predictedValue);
		}
		return confusionMatrixData;
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

	private static double getLogisticRegressionValue(double predictedValue) {
		return 1/(1+ Math.exp(-predictedValue));
	}	

}