/**
 * 
 */
package src.com.ml.hw3.classifier.stats;

import src.com.ml.hw3.data.Data;

/**
 * @author kkumar
 *
 */
public class NaiveBayesHistogram4BinsModel {
	
	private int bucketSize;
	private int featureSize;
	private int spamCount;
	private int nonSpamCount;
	private double probabilityOfSpam;
	private double[][] featureBucketValues;
	private double[][] featureBucketProbabilitiesSpam;
	double[][] featureBucketProbabilitiesNonSpam;
	
	public NaiveBayesHistogram4BinsModel(int featureSize, int spamCount, int nonSpamCount, int bucketSize) {
		this.bucketSize = bucketSize;
		this.featureSize = featureSize;
		this.spamCount = spamCount;
		this.nonSpamCount = nonSpamCount;
		this.probabilityOfSpam = (double) spamCount/(spamCount + nonSpamCount);
	}

	/**
	 * @return the spamCount
	 */
	public int getSpamCount() {
		return spamCount;
	}

	/**
	 * @param spamCount the spamCount to set
	 */
	public void setSpamCount(int spamCount) {
		this.spamCount = spamCount;
	}

	/**
	 * @return the nonSpamCount
	 */
	public int getNonSpamCount() {
		return nonSpamCount;
	}

	/**
	 * @param nonSpamCount the nonSpamCount to set
	 */
	public void setNonSpamCount(int nonSpamCount) {
		this.nonSpamCount = nonSpamCount;
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
	 * @return the featureBucketValues
	 */
	public double[][] getFeatureBucketValues() {
		return featureBucketValues;
	}

	/**
	 * @param featureBucketValues the featureBucketValues to set
	 */
	public void setFeatureBucketValues(double[][] featureBucketValues) {
		this.featureBucketValues = featureBucketValues;
	}

	/**
	 * @return the featureBucketProbabilitiesSpam
	 */
	public double[][] getFeatureBucketProbabilitiesSpam() {
		return featureBucketProbabilitiesSpam;
	}

	/**
	 * @param featureBucketProbabilitiesSpam the featureBucketProbabilitiesSpam to set
	 */
	public void setFeatureBucketProbabilitiesSpam(double[][] featureBucketProbabilitiesSpam) {
		applylaplaceSmoothing(featureBucketProbabilitiesSpam, this.spamCount);
		this.featureBucketProbabilitiesSpam = featureBucketProbabilitiesSpam;
	}

	/**
	 * @return the featureBucketProbabilitiesNonSpam
	 */
	public double[][] getFeatureBucketProbabilitiesNonSpam() {
		return featureBucketProbabilitiesNonSpam;
	}

	/**
	 * @param featureBucketProbabilitiesNonSpam the featureBucketProbabilitiesNonSpam to set
	 */
	public void setFeatureBucketProbabilitiesNonSpam(double[][] featureBucketProbabilitiesNonSpam) {
		applylaplaceSmoothing(featureBucketProbabilitiesNonSpam, this.nonSpamCount);
		this.featureBucketProbabilitiesNonSpam = featureBucketProbabilitiesNonSpam;
	}

	private void applylaplaceSmoothing(double[][] featureBucketProbabilities, int dataSize) {
		for(int feature = 0; feature < this.featureSize; feature++) {
			double[] bucketProbabilities = featureBucketProbabilities[feature];
			for(int col = 0; col < bucketProbabilities.length; col++) {
				bucketProbabilities[col] = (bucketProbabilities[col] + 1) / (dataSize + bucketSize);
			}
		}
	}

	public double calculateProbabilityOfSpam(Data dataPoint) {
		return calculateProbability(dataPoint, this.featureBucketValues, this.featureBucketProbabilitiesSpam, this.probabilityOfSpam);
	}

	public double calculateProbabilityOfNonSpam(Data dataPoint) {
		return calculateProbability(dataPoint, this.featureBucketValues, this.featureBucketProbabilitiesNonSpam, 1 - this.probabilityOfSpam);
	}
	
	private double calculateProbability(Data dataPoint, double[][] featureBucketValues, 
			double[][] featureBucketProbabilities, double probabilityClass) {
		double probability = 1;
		for(int feature = 0; feature < featureSize; feature++) {
			double featureValue = dataPoint.getFeatureValue(feature);
			double[] featureBucket = featureBucketValues[feature];
			double[] featureBucketProbability = featureBucketProbabilities[feature];
			int binforTestDataFeature = getBinForFeature(featureValue, featureBucket);
			probability*= featureBucketProbability[binforTestDataFeature];
		}
		probability*= probabilityClass;
		return probability;
	}

	private int getBinForFeature(double featureValue, double[] featureBucket) {
		int dataBin = -1;
		for(int bin=0; bin < featureBucket.length; bin++) {
			int nextBin = bin+1;
			if(bin == 0) {
				if(featureValue <= featureBucket[nextBin]) {
					dataBin = bin;
					break;
				}
			}
			if(bin == featureBucket.length -2) {
				if(featureValue > featureBucket[bin]) {
					dataBin = bin;
					break;
				}
			}
			if(featureValue > featureBucket[bin] && featureValue <= featureBucket[nextBin]) {
				dataBin = bin;
				break;
			}
		}
		return dataBin;
	}
}