package com.ml.hw4.feature;

import java.util.HashMap;
import java.util.Map;

public class LabelFeature extends Feature {

	Map<String, Double> labelValueMap;
	private int labelId;
	
	public LabelFeature(String featureName, int featureType) {
		super(featureName, featureType);
		labelValueMap = new HashMap<String, Double>();
	}
	
	public void addLabel(String label) {
		this.labelValueMap.put(label, (double) labelId);
		this.labelId++;
	}
}
