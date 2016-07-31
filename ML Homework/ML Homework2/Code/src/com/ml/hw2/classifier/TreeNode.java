package com.ml.hw2.classifier;

import java.util.List;

import com.ml.hw2.classifier.stats.NodeStats;
import com.ml.hw2.data.Data;
import com.ml.hw2.data.DataSet;
import com.ml.hw2.feature.Feature;

/**
 * @author Kamlendra Kumar
 *
 */
public class TreeNode {

	private static int INDEX = 0;
	private DataSet dataset;
	private TreeNode noBranch;
	private TreeNode yesBranch;
	private boolean isLeaf;
	private double leafValue;
	private NodeStats stats;
	private int splitFeatureIndex;
	private double splitFeatureValue;
	private int depth;
	private int nodeIndex;
	
	public TreeNode(DataSet dataset, int depth) throws Exception {
		this.dataset = dataset;
		noBranch = null;
		yesBranch = null;
		isLeaf = false;
		splitFeatureIndex = -1;
		splitFeatureValue = Double.MAX_VALUE;
		stats = createNodeStats(dataset);
		nodeIndex = INDEX++;
	}
	
	public TreeNode(int depth, List<Feature> features, int labelIndex, int classNum) {
		this.dataset = new DataSet(labelIndex, features);
		this.depth = depth;
		noBranch = null;
		yesBranch = null;
		isLeaf = false;
		splitFeatureIndex = -1;
		splitFeatureValue = Double.MAX_VALUE;
		this.dataset.setClassNum(classNum);
		stats = dataset.isClassificationTask() ? new NodeStats(dataset.classNum()) : new NodeStats();
		nodeIndex = INDEX++;
	}
	
	public void addData(Data data) throws Exception {
		this.dataset.addData(data);
		this.stats.add(data);
	}

	private NodeStats createNodeStats(DataSet dataset) throws Exception {
		NodeStats stats = dataset.isClassificationTask() ? new NodeStats(dataset.classNum()) : new NodeStats();
		for(Data data : dataset.getData()) {
			stats.add(data);
		}
		return stats;
	}
	
	public NodeStats getStats() {
		return stats;
	}
	
	public boolean isLeaf() {
		return isLeaf;
	}
	
	public void setLeaf() {
		isLeaf = true;
		leafValue = dataset.isClassificationTask() ? stats.predictClass() : stats.getMean();
	}
	
	public int depth() {
		return depth;
	}
	
	public void setDepth(int depth) {
		this.depth = depth;
	}
	
	public DataSet getDataSet() {
		return dataset;
	}

	public TreeNode getNoBranch() {
		return noBranch;
	}

	public void setNoBranch(TreeNode noBranch) {
		this.noBranch = noBranch;
	}

	public TreeNode getYesBranch() {
		return yesBranch;
	}

	public void setYesBranch(TreeNode yesBranch) {
		this.yesBranch = yesBranch;
	}

	public int getSplitFeatureIndex() {
		return splitFeatureIndex;
	}

	public void setSplitFeatureIndex(int splitFeatureIndex) {
		this.splitFeatureIndex = splitFeatureIndex;
	}

	public double getSplitFeatureValue() {
		return splitFeatureValue;
	}

	public void setSplitFeatureValue(double splitFeatureValue) {
		this.splitFeatureValue = splitFeatureValue;
	}

	public void setLeaf(boolean isLeaf) {
		this.isLeaf = isLeaf;
	}

	public int getNodeIndex() {
		return nodeIndex;
	}

	public void setNodeIndex(int nodeIndex) {
		this.nodeIndex = nodeIndex;
	}

	/**
	 * @return the leafValue
	 */
	public double getLeafValue() {
		return leafValue;
	}

	/**
	 * @param leafValue the leafValue to set
	 */
	public void setLeafValue(double leafValue) {
		this.leafValue = leafValue;
	}
}