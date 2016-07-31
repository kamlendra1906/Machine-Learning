/**
 * 
 */
package com.ml.hw4.feature;

/**
 * @author kkumar
 *
 */
public class DiscreteFeature extends Feature {

	private String discreteFeatureValue;
	
	public DiscreteFeature(String featureName, int featureType, String discreteFeatureValue) {
		super(featureName, featureType);
		this.discreteFeatureValue = discreteFeatureValue;
	}

	/**
	 * @return the discreteFeatureValue
	 */
	public String getDiscreteFeatureValue() {
		return discreteFeatureValue;
	}
	
	
}
