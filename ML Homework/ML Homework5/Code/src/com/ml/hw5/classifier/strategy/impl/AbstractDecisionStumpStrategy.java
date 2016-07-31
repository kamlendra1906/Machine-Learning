/**
 * 
 */
package com.ml.hw5.classifier.strategy.impl;

import com.ml.hw5.classifier.Classifier;
import com.ml.hw5.data.Data;
import com.ml.hw5.data.DataSet;

/**
 * @author kkumar
 *
 */
public abstract class AbstractDecisionStumpStrategy {

	protected double testFeatureThreshold(DataSet trainingDataSet, int featureIndex, double threshold, double[] dataErrorWeight) throws Exception {
		double error = 0;
		
		int counter = 0;
		for(Data data : trainingDataSet.getData()) {
			double actualLabel = data.labelValue() == 0 ? Classifier.NON_SPAM : Classifier.SPAM;
			double predictedLabel = data.getFeatureValue(featureIndex) <= threshold ? Classifier.NON_SPAM : Classifier.SPAM;
			if(actualLabel != predictedLabel) {
				error+= dataErrorWeight[counter];
			}
			counter++;
		}
		return error;
	}
}
