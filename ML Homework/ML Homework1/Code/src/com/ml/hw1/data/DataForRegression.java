/**
 * 
 */
package com.ml.hw1.data;

/**
 * @author Kamlendra Kumar
 *
 */
public class DataForRegression {

	private double[] featureData;
	private double[][] twoDArrayFeatureData;
	private double[] valueData;
	private int sampleSize;
	private int featureSize;
	
	/**
	 * @return the featureData
	 */
	public double[] getFeatureData() {
		return featureData;
	}
	/**
	 * @param featureData the featureData to set
	 */
	public void setFeatureData(double[] featureData) {
		this.featureData = featureData;
	}
	/**
	 * @return the valueData
	 */
	public double[] getValueData() {
		return valueData;
	}
	/**
	 * @param valueData the valueData to set
	 */
	public void setValueData(double[] valueData) {
		this.valueData = valueData;
	}
	/**
	 * @return the sampleSize
	 */
	public int getSampleSize() {
		return sampleSize;
	}
	/**
	 * @param sampleSize the sampleSize to set
	 */
	public void setSampleSize(int sampleSize) {
		this.sampleSize = sampleSize;
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
	 * @return the twoDArrayFeatureData
	 */
	public double[][] getTwoDArrayFeatureData() {
		return twoDArrayFeatureData;
	}
	/**
	 * @param twoDArrayFeatureData the twoDArrayFeatureData to set
	 */
	public void setTwoDArrayFeatureData(double[][] twoDArrayFeatureData) {
		this.twoDArrayFeatureData = twoDArrayFeatureData;
	}
}
