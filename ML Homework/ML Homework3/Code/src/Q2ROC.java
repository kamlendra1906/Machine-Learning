import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import src.com.ml.hw3.classifier.NaiveBayesBernouli;
import src.com.ml.hw3.classifier.NaiveBayesGaussianFeatureImpl;
import src.com.ml.hw3.classifier.NaiveBayesHistogram4Bins;
import src.com.ml.hw3.classifier.NaiveBayesHistogram9Bins;
import src.com.ml.hw3.classifier.stats.NaiveBayesGaussianFeatureModel;
import src.com.ml.hw3.classifier.stats.NaiveBayesHistogram4BinsModel;
import src.com.ml.hw3.classifier.stats.NaiveBayesModel;
import src.com.ml.hw3.data.DataInput;
import src.com.ml.hw3.data.DataSet;
import src.com.ml.hw3.util.ClassifierUtil;

/**
 * 
 */

/**
 * @author kkumar
 *
 */
public class Q2ROC {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		DataSet dataSet = DataInput.getData(ClassifierUtil.SPAM_TRAINING_FILE, ClassifierUtil.SPAM_FEATURES_FILE);

		int totalFolds = 10;
		int dataPerFold = dataSet.dataSize() / totalFolds;

		//Collections.shuffle(dataSet.getData());

		DataSet trainingData = null;
		DataSet testData = null;
		for (int fold = 0; fold < totalFolds; fold++) {
			trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			for (int counter = 0; counter < dataSet.dataSize(); counter++) {
				if (counter >= fold * dataPerFold && counter < (fold + 1) * dataPerFold) {
					testData.addData(dataSet.getData().get(counter));
				} else {
					trainingData.addData(dataSet.getData().get(counter));
				}
			}

			runNaiveBayes4Bin(trainingData, testData);

			if (fold == 0) {
				break;
			}
		}
	}

	public static void runNaiveBayes4Bin(DataSet trainingData, DataSet testData) throws Exception {
		int featureSize = trainingData.getFeatures().size() - 1;
		List<double[]> confusionMatrixData = new ArrayList<double[]>();

		NaiveBayesHistogram4Bins classifier = new NaiveBayesHistogram4Bins(featureSize, 4);
		NaiveBayesHistogram4BinsModel model = classifier.train(trainingData);

		List<Double> thresholds = classifier.getThresholds(model, testData);
		
		
		for (Double threshold : thresholds) {
			double[] confusionMatrix = new double[4];
			classifier.testModel(model, testData, confusionMatrix, threshold);
			confusionMatrixData.add(confusionMatrix);
		}
		List<double[]> dataPoints = ClassifierUtil.getROCCurveData(confusionMatrixData);
		for (double[] dataPoint : dataPoints) {
			System.out.println(ClassifierUtil.printArray(dataPoint));
		}
	}

	public static void runNaiveBayes9Bin(DataSet trainingData, DataSet testData) throws Exception {
		int featureSize = trainingData.getFeatures().size() - 1;
		List<double[]> confusionMatrixData = new ArrayList<double[]>();

		NaiveBayesHistogram9Bins classifier = new NaiveBayesHistogram9Bins(featureSize, 9);
		NaiveBayesHistogram4BinsModel model = classifier.train(trainingData);

		List<Double> thresholds = classifier.getThreshold(model, testData);
		for (Double threshold : thresholds) {
			double[] confusionMatrix = new double[4];
			classifier.testModel(model, testData, confusionMatrix, threshold);
			confusionMatrixData.add(confusionMatrix);
		}
		List<double[]> dataPoints = ClassifierUtil.getROCCurveData(confusionMatrixData);
		for (double[] dataPoint : dataPoints) {
			System.out.println(ClassifierUtil.printArray(dataPoint));
		}
	}

	public static void runNaiveBayesBernouli(DataSet trainingData, DataSet testData) throws Exception {

		List<double[]> confusionMatrixData = new ArrayList<double[]>();

		NaiveBayesBernouli classifier = new NaiveBayesBernouli(trainingData.getFeatures().size() - 1);
		NaiveBayesModel model = classifier.train(trainingData);

		List<Double> thresholds = classifier.getThresholds(model, testData);
		for (Double threshold : thresholds) {
			double[] confusionMatrix = new double[4];
			classifier.testModel(model, testData, confusionMatrix, threshold);
			confusionMatrixData.add(confusionMatrix);
		}
		List<double[]> dataPoints = ClassifierUtil.getROCCurveData(confusionMatrixData);
		for (double[] dataPoint : dataPoints) {
			System.out.println(ClassifierUtil.printArray(dataPoint));
		}
	}

	public static void runNaiveBayesGaussian(DataSet trainingData, DataSet testData) throws Exception {
		int featureSize = trainingData.getFeatures().size() - 1;
		List<double[]> confusionMatrixData = new ArrayList<double[]>();

		NaiveBayesGaussianFeatureImpl classifier = new NaiveBayesGaussianFeatureImpl(featureSize);
		NaiveBayesGaussianFeatureModel model = classifier.train(trainingData);

		List<Double> thresholds = classifier.getThresholds(model, testData);
		for (Double threshold : thresholds) {
			double[] confusionMatrix = new double[4];
			classifier.testModel(model, testData, confusionMatrix, threshold);
			confusionMatrixData.add(confusionMatrix);
		}
		List<double[]> dataPoints = ClassifierUtil.getROCCurveData(confusionMatrixData);
		for (double[] dataPoint : dataPoints) {
			System.out.println(ClassifierUtil.printArray(dataPoint));
		}
	}
}