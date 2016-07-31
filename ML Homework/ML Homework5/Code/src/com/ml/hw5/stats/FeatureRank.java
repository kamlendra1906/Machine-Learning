/**
 * 
 */
package com.ml.hw5.stats;

/**
 * @author kkumar
 *
 */
public class FeatureRank implements Comparable<FeatureRank>{
	
	private int feature;
	private double gamma;
	
	public FeatureRank() {
	}
	
	public FeatureRank(int feature, double gamma) {
		this.feature = feature;
		this.gamma = gamma;
	}
	
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
	 * @return the gamma
	 */
	public double getGamma() {
		return gamma;
	}
	/**
	 * @param gamma the gamma to set
	 */
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	@Override
	public int compareTo(FeatureRank featureRank) {
		if(this.gamma == featureRank.getGamma()) {
			return 0;
		}
		if(this.gamma > featureRank.getGamma()) {
			return -1;
		}
		return 1;
	}
}