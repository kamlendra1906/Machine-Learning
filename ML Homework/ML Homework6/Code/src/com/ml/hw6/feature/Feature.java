package com.ml.hw6.feature;

public class Feature implements IFeature{
	
	private final String featureName;
	private final int featureType;
	
	public Feature(String featureName, int featureType) {
		this.featureName = featureName;
		this.featureType = featureType;
	}
	
	public boolean isNumerical() {
		return this.featureType == NUMERICAL;
	}
	
	public boolean isDiscrete() {
		return this.featureType == DISCRETE;
	}
	
	public boolean isLabel() {
		return this.featureType == LABEL;
	}
	
	public boolean isNominal() {
		return this.featureType == NOMINAL;
	}
	
	public String getFeatureName() {
		return this.featureName;
	}
}