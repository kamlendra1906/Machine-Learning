/**
 * 
 */
package com.ml.hw7.stats;

/**
 * @author kkumar
 *
 */
public class NaiveBayesModel {

	private int featureSize;
	private double probabilityOfSpam;
	private double probabilityOfNonSpam;
	private double[] probFLessThanMeanSpam;
	private double[] probFGreaterThanMeanSpam;
	private double[] probFLessThanMeanNonSpam;
	private double[] probFGreaterThanMeanNonSpam;
	
	public NaiveBayesModel(int featureSize) {
		this.featureSize = featureSize;
		probFLessThanMeanSpam = new double[featureSize];
		probFGreaterThanMeanSpam = new double[featureSize];
		probFLessThanMeanNonSpam = new double[featureSize];
		probFGreaterThanMeanNonSpam = new double[featureSize];
	}
	
	/**
	 * @return the probFLessThanMeanSpam
	 */
	public double[] getProbFLessThanMeanSpam() {
		return probFLessThanMeanSpam;
	}
	/**
	 * @param probFLessThanMeanSpam the probFLessThanMeanSpam to set
	 */
	public void setProbFLessThanMeanSpam(int[] countFLessThanMeanSpam, int spamCount) {
		calculateProbability(this.probFLessThanMeanSpam, countFLessThanMeanSpam, spamCount);
	}
	
	/**
	 * @return the probFGreaterThanMeanSpam
	 */
	public double[] getProbFGreaterThanMeanSpam() {
		return probFGreaterThanMeanSpam;
	}
	/**
	 * @param probFGreaterThanMeanSpam the probFGreaterThanMeanSpam to set
	 */
	public void setProbFGreaterThanMeanSpam(int[] countFGreaterThanMeanSpam, int spamCount) {
		calculateProbability(this.probFGreaterThanMeanSpam, countFGreaterThanMeanSpam, spamCount);
	}
	/**
	 * @return the probFLessThanMeanNonSpam
	 */
	public double[] getProbFLessThanMeanNonSpam() {
		return probFLessThanMeanNonSpam;
	}
	/**
	 * @param probFLessThanMeanNonSpam the probFLessThanMeanNonSpam to set
	 */
	public void setProbFLessThanMeanNonSpam(int[] countFLessThanMeanNonSpam, int nonSpamCount) {
		calculateProbability(this.probFLessThanMeanNonSpam, countFLessThanMeanNonSpam, nonSpamCount);
	}
	/**
	 * @return the probFGreaterThanMeanNonSpam
	 */
	public double[] getProbFGreaterThanMeanNonSpam() {
		return probFGreaterThanMeanNonSpam;
	}
	/**
	 * @param probFGreaterThanMeanNonSpam the probFGreaterThanMeanNonSpam to set
	 */
	public void setProbFGreaterThanMeanNonSpam(int[] countFGreaterThanMeanNonSpam, int nonSpamCount) {
		calculateProbability(this.probFGreaterThanMeanNonSpam, countFGreaterThanMeanNonSpam, nonSpamCount);
	}
	
	/**
	 * @return the probabilityOfSpam
	 */
	public double getProbabilityOfSpam() {
		return probabilityOfSpam;
	}

	/**
	 * @param probabilityOfSpam the probabilityOfSpam to set
	 */
	public void setProbabilityOfSpam(double probabilityOfSpam) {
		this.probabilityOfSpam = probabilityOfSpam;
	}

	/**
	 * @return the probabilityOfNonSpam
	 */
	public double getProbabilityOfNonSpam() {
		return probabilityOfNonSpam;
	}

	/**
	 * @param probabilityOfNonSpam the probabilityOfNonSpam to set
	 */
	public void setProbabilityOfNonSpam(double probabilityOfNonSpam) {
		this.probabilityOfNonSpam = probabilityOfNonSpam;
	}

	private void calculateProbability(double[] probArray, int[] countArray, int count) {
		for(int featureIndex = 0; featureIndex < this.featureSize; featureIndex++) {
			probArray[featureIndex] = (double)countArray[featureIndex] / count;
		}
	}

}