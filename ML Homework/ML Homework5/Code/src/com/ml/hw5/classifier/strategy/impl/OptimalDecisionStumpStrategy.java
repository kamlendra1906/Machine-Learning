/**
 * 
 */
package com.ml.hw5.classifier.strategy.impl;

import java.util.Map;
import java.util.Set;

import com.ml.hw5.classifier.impl.AdaBoost;
import com.ml.hw5.classifier.strategy.DecisionStumpStrategy;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.model.DecisionStumpModel;

/**
 * @author kkumar
 *
 */
public class OptimalDecisionStumpStrategy extends AbstractDecisionStumpStrategy implements DecisionStumpStrategy {

	@SuppressWarnings("unchecked")
	@Override
	public DecisionStumpModel getDecisionStump(DataSet trainingDataSet, Map<String, Object> additionalData) throws Exception {
		Map<Integer, Set<Double>> featureIdValueMap = (Map<Integer, Set<Double>>) additionalData.get("hi");
		DecisionStumpModel model = findBestDecisionStump(trainingDataSet, featureIdValueMap, additionalData);
		return model;
	}
	
	private DecisionStumpModel findBestDecisionStump(DataSet trainingDataSet, Map<Integer, Set<Double>> featureIdValueMap, Map<String, Object> additionalData) throws Exception {
		int featureSize = trainingDataSet.getFeatures().size() - 1;
		
		int bestFeatureIndex = -1;
		double bestMaxDistanceFromHalf = Double.NEGATIVE_INFINITY;
		double bestFeatureThreshold = -1;
		double bestError = 0;
		
		double[] dataErrorWeight = (double[]) additionalData.get(AdaBoost.DATA_ERROR_WEIGHT);
		
		for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			Set<Double> featureValues = featureIdValueMap.get(featureIndex);
			Double[] featureValuesArray = featureValues.toArray(new Double[featureValues.size()]);
			for(int index = 0; index < featureValues.size() - 1; index++) {
				int nextIndex = index + 1;
				double threshold = (featureValuesArray[index] + featureValuesArray[nextIndex])/2;
				double error = this.testFeatureThreshold(trainingDataSet, featureIndex, threshold, dataErrorWeight);
				if(Math.abs(0.5 - error) > bestMaxDistanceFromHalf) {
					bestFeatureIndex = featureIndex;
					bestMaxDistanceFromHalf = Math.abs(0.5 - error);
					bestError = error;
					bestFeatureThreshold = threshold;
				}
			}
		}
		
		DecisionStumpModel model = new DecisionStumpModel();
		model.setFeature(bestFeatureIndex);
		model.setThreshold(bestFeatureThreshold);
		model.setRoundError(bestError);
		return model;
	}
}