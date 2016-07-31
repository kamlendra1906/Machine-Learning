/**
 * 
 */
package com.ml.hw7.classifier.strategy.impl;

import java.util.Map;
import java.util.Random;

import com.ml.hw7.classifier.impl.AdaBoost;
import com.ml.hw7.classifier.strategy.DecisionStumpStrategy;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.model.DecisionStumpModel;

/**
 * @author kkumar
 *
 */
public class RandomDecisionStumpStrategy extends AbstractDecisionStumpStrategy implements DecisionStumpStrategy {

	@Override
	public DecisionStumpModel getDecisionStump(DataSet trainingDataSet, Map<String, Object> additionalData) throws Exception {
		int featureSize = trainingDataSet.getFeatures().size() - 1;
		double[] featureMinValues = trainingDataSet.getFeatureMin();
		double[] featureMaxValues = trainingDataSet.getFeatureMax();
		double[] dataErrorWeight = (double[]) additionalData.get(AdaBoost.DATA_ERROR_WEIGHT);
		
		int randomFeature =  (int) getRandomValue(0, featureSize -1);
		double threshold = this.getRandomValue(featureMinValues[randomFeature], featureMaxValues[randomFeature]);
		double roundError = this.testFeatureThreshold(trainingDataSet, randomFeature, threshold, dataErrorWeight);
		
		double[] array = new double[4];
		array[0] = randomFeature;
		array[1] = featureMinValues[randomFeature];
		array[2] = threshold;
		array[3] = featureMaxValues[randomFeature];
		
		//System.out.println(ClassifierUtil.printArray(array));
		
		DecisionStumpModel model = new DecisionStumpModel();
		model.setFeature(randomFeature);
		model.setRoundError(roundError);
		model.setThreshold(threshold);
		return model;
	}

	private double getRandomValue(double start, double end) {
		Random random = new Random();
		double range = end - start;
		double fraction = range * random.nextDouble();
		return fraction + start;
	}
}