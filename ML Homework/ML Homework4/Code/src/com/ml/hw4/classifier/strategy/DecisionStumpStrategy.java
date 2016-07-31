/**
 * 
 */
package com.ml.hw4.classifier.strategy;

import java.util.Map;

import com.ml.hw4.data.DataSet;
import com.ml.hw4.model.DecisionStumpModel;

/**
 * @author kkumar
 *
 */
public interface DecisionStumpStrategy {

	public DecisionStumpModel getDecisionStump(DataSet trainingDataSet, Map<String, Object> additionalData) throws Exception;

}
