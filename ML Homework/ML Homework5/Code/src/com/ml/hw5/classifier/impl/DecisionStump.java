/**
 * 
 */
package com.ml.hw5.classifier.impl;

import java.util.Map;

import com.ml.hw5.classifier.Classifier;
import com.ml.hw5.classifier.strategy.DecisionStumpStrategy;
import com.ml.hw5.classifier.strategy.impl.OptimalDecisionStumpStrategy;
import com.ml.hw5.classifier.strategy.impl.RandomDecisionStumpStrategy;
import com.ml.hw5.data.Data;
import com.ml.hw5.data.DataSet;
import com.ml.hw5.model.DecisionStumpModel;

/**
 * @author kkumar
 *
 */
public class DecisionStump implements Classifier {

	private DecisionStumpModel model;
	private DecisionStumpStrategy strategy;
	
	public DecisionStump(boolean optimalDecisionStump) {
		strategy = optimalDecisionStump ? new OptimalDecisionStumpStrategy() : new RandomDecisionStumpStrategy();
	}
	
	@Override
	public void trainModel(DataSet trainingDataSet, DataSet testDataSet,  Map<String, Object> additionalData) throws Exception {
		model = strategy.getDecisionStump(trainingDataSet, additionalData);
		
	}

	@Override
	public double testModel(DataSet testDataSet, Map<String, Object> additionalData) throws Exception {
		int error = 0;
		for(Data data : testDataSet.getData()) {
			double actualLabel = data.labelValue() == 0 ? Classifier.NON_SPAM : Classifier.SPAM;
			double predictedLabel = classifyTestPoint(data, null);
			if(actualLabel != predictedLabel) {
				error+= 1;
			}
		}
		return (double) error / testDataSet.dataSize();
	}

	@Override
	public double classifyTestPoint(Data dataPoint, Map<String, Object> additionalData) throws Exception {
		int featureIndex = model.getFeature();
		double threshold = model.getThreshold();
		return dataPoint.getFeatureValue(featureIndex) <= threshold ? Classifier.NON_SPAM : Classifier.SPAM;
	}

	/**
	 * @return the model
	 */
	public DecisionStumpModel getModel() {
		return model;
	}

	/**
	 * @param model the model to set
	 */
	public void setModel(DecisionStumpModel model) {
		this.model = model;
	}
	
	@Override
	public String toString() {
		return "feature = "+model.getFeature()+"   threshold = "+model.getThreshold()+"   round error = "+model.getRoundError(); 
	}
}