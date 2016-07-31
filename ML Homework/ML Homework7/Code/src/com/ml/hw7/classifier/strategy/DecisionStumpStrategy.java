/**
 * 
 */
package com.ml.hw7.classifier.strategy;

import java.util.Map;

import com.ml.hw7.data.DataSet;
import com.ml.hw7.model.DecisionStumpModel;

/**
 * @author kkumar
 *
 */
public interface DecisionStumpStrategy {

	public DecisionStumpModel getDecisionStump(DataSet trainingDataSet, Map<String, Object> additionalData) throws Exception;

}
