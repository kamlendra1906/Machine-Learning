/**
 * 
 */
package com.ml.hw7.model;

import org.apache.commons.math3.analysis.function.Gaussian;

import com.ml.hw7.data.Data;
import com.ml.hw7.util.ClassifierUtil;

/**
 * @author kkumar
 *
 */
public class NaiveBayesGaussianFeatureModel {

	private int featureSize;
	private double[] featureMeanSpam;
	private double[] featureMeanNonSpam;
	private double[] featureVariance;
	private double probabilityOfSpam;
	
	public NaiveBayesGaussianFeatureModel(int featureSize) {
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
	 * @return the featureVarianceNonSpam
	 */
	public double[] getFeatureVariance() {
		return featureVariance;
	}

	/**
	 * @param featureVarianceNonSpam the featureVarianceNonSpam to set
	 */
	public void setFeatureVariance(double[] featureVariance) {
		this.featureVariance = featureVariance;
	}

	/**
	 * @return the probabilityOfOne
	 */
	public double getProbabilityOfOne() {
		return probabilityOfSpam;
	}

	/**
	 * @param probabilityOfOne the probabilityOfOne to set
	 */
	public void setProbabilityOfSpam(double probabilityOfSpam) {
		this.probabilityOfSpam = probabilityOfSpam;
	}
	
	public double calculateProbabilityOfSpam(Data data) {
		return calculateProbabilityOfClass(data, this.featureMeanSpam, this.featureVariance, this.probabilityOfSpam);
	}

	public double calculateProbabilityOfNonSpam(Data data) {
		return calculateProbabilityOfClass(data, this.featureMeanNonSpam, this.featureVariance, 1 - this.probabilityOfSpam);	
	}
	
	private double calculateProbabilityOfClass(Data data, double[] featureMean, double[] featureVariance, double probabilityOfClass) {
		double probability = 1;
		for(int featureIndex = 0; featureIndex < this.featureSize; featureIndex++) {
			probability*= getFeatureProbabilityOfClass(data.getFeatureValue(featureIndex), featureMean[featureIndex], 
					featureVariance[featureIndex]);
		}
		return probability * probabilityOfClass;
	}
	
	private double calculateProbabilityOfClassByLogLikelihood(Data data, double[] featureMean, double[] featureVariance, double probabilityOfClass) {
		double probability = 0;
		for(int featureIndex = 0; featureIndex < this.featureSize; featureIndex++) {
			probability+= ClassifierUtil.logValue(getFeatureProbabilityOfClass(data.getFeatureValue(featureIndex), featureMean[featureIndex], 
					featureVariance[featureIndex]));
		}
		return probability + ClassifierUtil.logValue(probabilityOfClass);
	}

	private double getFeatureProbabilityOfClass(double featureValue, double mean, double varriance) {
		double standardDeviation = Math.sqrt(varriance == 0 ? 0.001 : varriance);
		Gaussian gaussian = new Gaussian(mean, standardDeviation);
		return gaussian.value(featureValue);
	}

}
