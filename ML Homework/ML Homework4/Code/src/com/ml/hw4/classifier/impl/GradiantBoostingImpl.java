/**
 * 
 */
package com.ml.hw4.classifier.impl;

import java.util.ArrayList;
import java.util.List;

import com.ml.hw4.classifier.bagging.DecisionClassifier;
import com.ml.hw4.classifier.bagging.DecisionClassifierImpl;
import com.ml.hw4.classifier.bagging.TreeNode;
import com.ml.hw4.data.Data;
import com.ml.hw4.data.DataSet;

/**
 * @author kkumar
 *
 */
public class GradiantBoostingImpl {

	private List<TreeNode> regressonTreeRoots;
	private int boostingRound;
	
	public GradiantBoostingImpl(int boostingRound) {
		this.boostingRound = boostingRound;
		regressonTreeRoots = new ArrayList<TreeNode>();
	}
	
	public void trainModel(DataSet trainingDataSet) throws Exception {
		for(int round = 0; round < this.boostingRound; round++) {
			DecisionClassifier regressor = new DecisionClassifierImpl(2, .15, 40);
			TreeNode root = regressor.buildClassifier(trainingDataSet);
			regressonTreeRoots.add(root);
			testModelAndUpdateInput(root, trainingDataSet);
		}
	}

	private void testModelAndUpdateInput(TreeNode treeRoot, DataSet trainingDataSet) throws Exception {
		int labelValueIndex = trainingDataSet.getLabelIndex();
		for(Data data : trainingDataSet.getData()) {
			double actualValue = data.labelValue();
			double predictedValue = treeRoot.predict(data);
			double gradiant = actualValue - predictedValue;
			data.setFeatureValue(labelValueIndex, gradiant);
		}
	}
	
	private double predictValue(Data data) throws Exception {
		double predictedValue = 0;
		for(TreeNode root : regressonTreeRoots) {
			predictedValue+= root.predict(data);
		}
		return predictedValue;
	}
	
	public double testModel(DataSet testDataSet) throws Exception {
		double diffSum = 0;
		for(Data data : testDataSet.getData()){
			double perdictedValue = this.predictValue(data);
			diffSum += Math.pow((data.labelValue() - perdictedValue), 2);
		}
		double rmsd = diffSum/testDataSet.getData().size();
		return rmsd;
	}
}
