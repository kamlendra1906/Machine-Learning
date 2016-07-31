/**
 * 
 */
package com.ml.hw5.classifier.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.ml.hw5.data.Data;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.stats.NaiveBayesModel;
import com.ml.hw5.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class NaiveBayesBernouli {

	public static final double SPAM = 1.0d;
	public static final double NON_SPAM = 0.0d;

	private int featureSize;
	private int[] fLessThanMeanSpam;
	private int[] fGreaterThanMeanSpam;
	private int[] fLessThanMeanNonSpam;
	private int[] fGreaterThanMeanNonSpam;
	private double[] featureMean;
	
	public NaiveBayesBernouli(int featureSize) {
		this.featureSize = featureSize;
		fLessThanMeanSpam = new int[featureSize];
		fGreaterThanMeanSpam = new int[featureSize];
		fLessThanMeanNonSpam = new int[featureSize];
		fGreaterThanMeanNonSpam = new int[featureSize];
	}

	public NaiveBayesModel train(DataSet trainingDataSet) throws Exception {
		int dataSize = trainingDataSet.dataSize();
		featureMean = trainingDataSet.getDataFeatureMean();
		
		int spamCount = 0;
		int nonSpamCount = 0;
		
		for(Data data : trainingDataSet.getData()) {
			double classLabel = data.labelValue();
			if(classLabel == SPAM) {
				spamCount++;
				updateStatsForSpam(data, featureMean);
			} else {
				nonSpamCount++;
				updateSpamCountForNonSpam(data, featureMean);
			}
		}
		
		NaiveBayesModel model = new NaiveBayesModel(featureSize);
		model.setProbabilityOfSpam((double) spamCount/dataSize);
		model.setProbabilityOfNonSpam((double)nonSpamCount/dataSize);
		model.setProbFLessThanMeanSpam(fLessThanMeanSpam, spamCount);
		model.setProbFGreaterThanMeanSpam(fGreaterThanMeanSpam, spamCount);
		model.setProbFLessThanMeanNonSpam(fLessThanMeanNonSpam, nonSpamCount);
		model.setProbFGreaterThanMeanNonSpam(fGreaterThanMeanNonSpam, nonSpamCount);
		return model;
	}

	private void updateStatsForSpam(Data data, double[] featureMean) {
		for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			double featureValue = data.getFeatureValue(featureIndex);
			if(Double.isNaN(featureValue)) {
				continue;
			}
			if(featureValue <= featureMean[featureIndex]) {
				fLessThanMeanSpam[featureIndex]++;
			} else {
				fGreaterThanMeanSpam[featureIndex]++;
			}
		}
	}
	
	private void updateSpamCountForNonSpam(Data data, double[] featureMean) {
		for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			double featureValue = data.getFeatureValue(featureIndex);
			if(Double.isNaN(featureValue)) {
				continue;
			}
			if(featureValue <= featureMean[featureIndex]) {
				fLessThanMeanNonSpam[featureIndex]++;
			} else {
				fGreaterThanMeanNonSpam[featureIndex]++;
			}
		}
	}

	public double testModel(NaiveBayesModel model, DataSet testData, double[] confusionMatrix, double threshold) throws Exception {
		
		double totalError = 0;

		for (Data dataPoint : testData.getData()) {
			double actualClass = dataPoint.labelValue();
			double predictedClass = predictClass(model, dataPoint, threshold);
			ClassifierUtil.updateConfusionMatrix(confusionMatrix, actualClass, predictedClass);
			if (actualClass != predictedClass) {
				totalError++;
			}
		}
		return totalError / testData.dataSize();

	}

	private double predictClass(NaiveBayesModel model, Data dataPoint, double threshold) {
		double probabilityOfSpam = calculateProbabilityOfSpam(model, dataPoint);
		double probabilityOfNonSpam = calculateProbabilityOfNonSpam(model, dataPoint);
		
		/*if(probabilityOfSpam/probabilityOfNonSpam  > threshold) {
			return SPAM;
		}
		return NON_SPAM;*/
		if(probabilityOfSpam >= probabilityOfNonSpam) {
			return SPAM;
		}
		return NON_SPAM;
	}
	
	public List<Double> getThresholds(NaiveBayesModel model, DataSet testData) {
		List<Double> thresholds = new ArrayList<Double>();
		for(Data data : testData.getData()) {
			double probabilityOfSpam = calculateProbabilityOfSpam(model, data);
			double probabilityOfNonSpam = calculateProbabilityOfNonSpam(model, data);
			thresholds.add(probabilityOfSpam / probabilityOfNonSpam);
		}
		Collections.sort(thresholds);
		return thresholds;
	}

	private double calculateProbabilityOfSpam(NaiveBayesModel model, Data dataPoint) {
		double probability = 1;
		for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			double featureValue = dataPoint.getFeatureValue(featureIndex);
			
			if(Double.isNaN(featureValue)) {
				continue;
			}
			if(featureValue <= featureMean[featureIndex]) {
				probability*= model.getProbFLessThanMeanSpam()[featureIndex];
			} else {
				probability*= model.getProbFGreaterThanMeanSpam()[featureIndex];
			}
		}
		probability*= model.getProbabilityOfSpam();
		return probability;
	}
	
	private double calculateProbabilityOfNonSpam(NaiveBayesModel model, Data dataPoint) {
		double probability = 1;
		for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			
			double featureValue = dataPoint.getFeatureValue(featureIndex);
			
			if(Double.isNaN(featureValue)) {
				continue;
			}
			
			if(featureValue <= featureMean[featureIndex]) {
				probability*= model.getProbFLessThanMeanNonSpam()[featureIndex];
			} else {
				probability*= model.getProbFGreaterThanMeanNonSpam()[featureIndex];
			}
		}
		probability*= model.getProbabilityOfNonSpam();
		return probability;
	}
}