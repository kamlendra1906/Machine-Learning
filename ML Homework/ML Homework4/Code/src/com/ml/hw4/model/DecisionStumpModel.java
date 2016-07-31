/**
 * 
 */
package com.ml.hw4.model;

/**
 * @author kkumar
 *
 */
public class DecisionStumpModel {

	private int feature;
	private double threshold;
	private double roundError;
	
	/**
	 * @return the feature
	 */
	public int getFeature() {
		return feature;
	}
	/**
	 * @param feature the feature to set
	 */
	public void setFeature(int feature) {
		this.feature = feature;
	}
	/**
	 * @return the threshold
	 */
	public double getThreshold() {
		return threshold;
	}
	/**
	 * @param threshold the threshold to set
	 */
	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}
	/**
	 * @return the roundError
	 */
	public double getRoundError() {
		return roundError;
	}
	/**
	 * @param roundError the roundError to set
	 */
	public void setRoundError(double roundError) {
		this.roundError = roundError;
	}
}