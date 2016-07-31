package com.ml.hw4.feature;

public class Feature implements IFeature{

	public static final int NUMERICAL = 1;
	public static final int NOMINAL = 2;
	public static final int DISCRETE = 3;
	public static final int LABEL = 4;
	
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