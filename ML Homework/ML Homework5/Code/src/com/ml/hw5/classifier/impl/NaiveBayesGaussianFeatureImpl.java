/**
 * 
 */
package com.ml.hw5.classifier.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.ml.hw5.data.Data;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.model.NaiveBayesGaussianFeatureModel;
import com.ml.hw5.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class NaiveBayesGaussianFeatureImpl {

	public static final double SPAM = 1.0d;
	public static final double NON_SPAM = 0.0d;

	private int featureSize;
	private double[] featureMeanSpam;
	private double[] featureMeanNonSpam;
	private double[] featureVarriance;
	
	public NaiveBayesGaussianFeatureImpl(int featureSize) {
		this.featureSize = featureSize;
		this.featureMeanSpam = new double[featureSize];
		this.featureMeanNonSpam = new double[featureSize];
		this.featureVarriance = new double[featureSize];
	}

	public NaiveBayesGaussianFeatureModel train(DataSet trainingDataSet) throws Exception {
		
		double[] featureMean = trainingDataSet.getDataFeatureMean();
		this.featureVarriance = this.calculateFeatureVarriance(trainingDataSet, featureMean);
		
		DataSet spamDataSet = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		DataSet nonSPamDataSet = new DataSet(trainingDataSet.getLabelIndex(), trainingDataSet.getFeatures());
		
		for(Data data : trainingDataSet.getData()) {
			double classLabel = data.labelValue();
			if(classLabel == SPAM) {
				spamDataSet.addData(data);
			} else {
				nonSPamDataSet.addData(data);
			}
		}
		
		this.featureMeanSpam = spamDataSet.getDataFeatureMean();
		this.featureMeanNonSpam = nonSPamDataSet.getDataFeatureMean();
		
		NaiveBayesGaussianFeatureModel model = new NaiveBayesGaussianFeatureModel(this.featureSize);
		model.setFeatureMeanSpam(this.featureMeanSpam);
		model.setFeatureMeanNonSpam(this.featureMeanNonSpam);
		model.setFeatureVariance(this.featureVarriance);
		model.setProbabilityOfSpam((double) spamDataSet.dataSize() / trainingDataSet.dataSize());
		return model;
	}
	
	public double[] calculateFeatureVarriance(DataSet dataSet, double[] featureMean) throws Exception {
		double[] featureVarriance = new double[this.featureSize];
		int dataSize = dataSet.dataSize();
		for(Data data : dataSet.getData()) {
			for(int featureIndex = 0; featureIndex < this.featureSize; featureIndex++) {
				featureVarriance[featureIndex]+= Math.pow(data.getFeatureValue(featureIndex) - featureMean[featureIndex], 2);
			}
		}
		for(int featureIndex = 0; featureIndex < this.featureSize; featureIndex++) {
			featureVarriance[featureIndex] =  featureVarriance[featureIndex] / dataSize;
		}
		return featureVarriance;
	}
	
	public double testModel(NaiveBayesGaussianFeatureModel model, DataSet testData, double[] confusionMatrix, double threshold)  
			throws Exception {
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
	

	private double predictClass(NaiveBayesGaussianFeatureModel model, Data dataPoint, double threshold) {
		double probabilityOfSpam = model.calculateProbabilityOfSpam(dataPoint);
		double probabilityOfNonSpam = model.calculateProbabilityOfNonSpam(dataPoint);
		
		if(probabilityOfSpam/probabilityOfNonSpam > threshold) {
			return SPAM;
		}
		return NON_SPAM;
		/*if(probabilityOfNonSpam >= probabilityOfSpam) {
			return NON_SPAM;
		}
		return SPAM;
*/
	}
	
	public List<Double> getThresholds(NaiveBayesGaussianFeatureModel model, DataSet testData) {
		List<Double> thresholds = new ArrayList<Double>();
		for (Data dataPoint : testData.getData()) {
			double probabilityOfSpam = model.calculateProbabilityOfSpam(dataPoint);
			double probabilityOfNonSpam = model.calculateProbabilityOfNonSpam(dataPoint);
			thresholds.add(probabilityOfSpam/probabilityOfNonSpam);
		}
		Collections.sort(thresholds);
		return thresholds;
	}

	
	/**
	 * @return the featureSize
	 */
	public int getFeatureSize() {
		return featureSize;
	}

	/**
	 * @param featureSize the featureSize to set
	 */
	public void setFeatureSize(int featureSize) {
		this.featureSize = featureSize;
	}

	/**
	 * @return the featureMeanSpam
	 */
	public double[] getFeatureMeanSpam() {
		return featureMeanSpam;
	}

	/**
	 * @param featureMeanSpam the featureMeanSpam to set
	 */
	public void setFeatureMeanSpam(double[] featureMeanSpam) {
		this.featureMeanSpam = featureMeanSpam;
	}

	/**
	 * @return the featureMeanNonSpam
	 */
	public double[] getFeatureMeanNonSpam() {
		return featureMeanNonSpam;
	}

	/**
	 * @param featureMeanNonSpam the featureMeanNonSpam to set
	 */
	public void setFeatureMeanNonSpam(double[] featureMeanNonSpam) {
		this.featureMeanNonSpam = featureMeanNonSpam;
	}

	/**
	 * @return the featureVarrianceNonSpam
	 */
	public double[] getFeatureVarriance() {
		return featureVarriance;
	}

	/**
	 * @param featureVarrianceNonSpam the featureVarrianceNonSpam to set
	 */
	public void setFeatureVarriance(double[] featureVarrianceNonSpam) {
		this.featureVarriance = featureVarrianceNonSpam;
	}
}
