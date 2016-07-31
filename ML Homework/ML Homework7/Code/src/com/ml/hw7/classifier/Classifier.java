/**
 * 
 */
package com.ml.hw7.classifier;

import java.util.Map;

import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;

/**
 * @author Kamlendra
 *
 */
public interface Classifier {

	public static final int SPAM = 1;
	public static final int NON_SPAM = -1;
	public static final String SVM_PARAMETERS = "SVM_PARAMETERS";
	
	/**
	 * Trains the model.
	 * @param trainingDataSet
	 * @param additionalData
	 * @throws Exception
	 */
	public void trainModel(DataSet trainingDataSet, DataSet testDataSet, Map<String, Object> additionalData) throws Exception;
	
	/**
	 * Tests the model.
	 * @param testDataSet
	 * @param additionalData
	 * @return
	 * @throws Exception
	 */
	public double testModel(DataSet testDataSet, Map<String, Object> additionalData) throws Exception;
	
	/**
	 * Classifies the testPoint
	 * @param dataPoint
	 * @param additionalData
	 * @return
	 * @throws Exception
	 */
	public double classifyTestPoint(Data dataPoint, Map<String, Object> additionalData) throws Exception;
}
