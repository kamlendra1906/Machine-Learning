/**
 * 
 */
package com.ml.hw2.classifier;

import com.ml.hw2.data.Data;
import com.ml.hw2.data.DataSet;

/**
 * @author kkumar
 *
 */
public interface DecisionClassifier {

	/**
	 * Creates the decision tree model
	 * @param dataset
	 * @return
	 * @throws Exception
	 */
	public TreeNode buildClassifier(DataSet dataset) throws Exception;
	
	/**
	 * Returns the class of the data point.
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double predict(TreeNode node, Data data) throws Exception;
}
