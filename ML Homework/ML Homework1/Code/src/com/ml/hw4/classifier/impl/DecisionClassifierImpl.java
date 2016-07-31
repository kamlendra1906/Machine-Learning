/**
 * 
 */
package com.ml.hw4.classifier.impl;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import com.ml.hw1.classifier.DecisionClassifier;
import com.ml.hw1.classifier.EntropyMeasure;
import com.ml.hw1.classifier.SplitNodeResult;
import com.ml.hw1.classifier.TreeBuilder;
import com.ml.hw1.classifier.TreeNode;
import com.ml.hw1.classifier.stats.NodeStats;
import com.ml.hw1.classifier.stats.NumericalFeatureThreshold;
import com.ml.hw1.data.Data;
import com.ml.hw1.data.DataSet;

/**
 * @author kkumar
 *
 */
public class DecisionClassifierImpl implements DecisionClassifier, TreeBuilder {
	
	/* (non-Javadoc)
	 * @see com.ml.hw1.classifier.DecisionClassifier#buildClassifier(com.ml.hw1.data.DataSet)
	 */
	private int maxDepth;
	private double minGain;
	private int minDataPerNode;
	
	public DecisionClassifierImpl(int maxDepth, double minGain, int minDataPerNode) {
		this.maxDepth = maxDepth;
		this.minGain = minGain;
		this.minDataPerNode = minDataPerNode;
	}
	
	@Override
	public TreeNode buildClassifier(DataSet dataset) throws Exception {
		TreeNode root = new TreeNode(dataset, 0);
		buildTree(root);
		return root;
	}

	public TreeNode buildTree(TreeNode root) throws Exception {
		Queue<TreeNode> nodesToBeSplit = new ArrayDeque<TreeNode>();
		Map<Integer, Boolean> featureUsed = new HashMap<Integer, Boolean>();
		nodesToBeSplit.add(root);
		while(!nodesToBeSplit.isEmpty()) {
			splitNode(nodesToBeSplit.remove(), nodesToBeSplit, featureUsed);
		}
		return root;
	}
	
	private void splitNode(TreeNode node, Queue<TreeNode> nodesToBeSplit, Map<Integer, Boolean> featureUsed) throws Exception {
		NodeStats stats = node.getStats();
		DataSet dataSet = node.getDataSet();
		if(node.isLeaf() || node.depth() == maxDepth || (dataSet.isClassificationTask() && isPureNode(stats)) || 
				stats.getTotalData() <= minDataPerNode) {
			node.setLeaf();
			return;
		}
		double maxInfoGain = Double.NEGATIVE_INFINITY;
		SplitNodeResult splitResult = null;
		SplitNodeResult bestResult = null;
		for(int i=0; i< dataSet.getFeatures().size(); i++) {
			if(i != dataSet.getLabelIndex()) {
				if(dataSet.getFeatures().get(i).isNominal() && !featureUsed.containsKey(i)) {
					splitResult = handleNominalFeature(i, node);
				} else {
					splitResult = handleNumericalFeature(i, node);
				}
				if(splitResult == null) {
					continue;
				}
				if(splitResult.getInfoGain() > maxInfoGain) {
					maxInfoGain = splitResult.getInfoGain();
					bestResult = splitResult;
				}
			}
		}
		if(bestResult == null || (bestResult != null && 
				bestResult.getInfoGain() < minGain)) {
			node.setLeaf();
			return;
		}
		featureUsed.put(bestResult.getSplitFeatureIndex(), true);
		updateSplitCriteriaOfNode(node, bestResult);
		nodesToBeSplit.add(bestResult.getNoBranch());
		nodesToBeSplit.add(bestResult.getYesBranch());
	}

	private boolean isPureNode(NodeStats stats) {
		int[] labelsPerClass = stats.getLabelPerClass();
		for(int labelCount : labelsPerClass) {
			if(labelCount == stats.getTotalData()) {
				return true;
			}
		}
		return false;
	}

	private void updateSplitCriteriaOfNode(TreeNode node, SplitNodeResult bestResult) throws Exception {
		node.setNoBranch(bestResult.getNoBranch());
		node.setYesBranch(bestResult.getYesBranch());
		node.setSplitFeatureIndex(bestResult.getSplitFeatureIndex());
		if(node.getDataSet().getFeature(bestResult.getSplitFeatureIndex()).isNumerical()) {
			node.setSplitFeatureValue(bestResult.getSplitFeatureValue());
		}
	}

	private SplitNodeResult handleNominalFeature(int featureIndex, TreeNode node) throws Exception {
		SplitNodeResult result = new SplitNodeResult();
		DataSet dataSet = node.getDataSet();
		double oldEntropy = EntropyMeasure.calculateRandomness(node);
		TreeNode noBranch = new TreeNode(node.depth()+1, dataSet.getFeatures(), node.getDataSet().getLabelIndex(), dataSet.classNum());
		TreeNode yesBranch = new TreeNode(node.depth()+1, dataSet.getFeatures(), node.getDataSet().getLabelIndex(), dataSet.classNum());
		for(Data data : dataSet.getData()) {
			if(data.getFeatureValue(featureIndex) == 0d) {
				noBranch.addData(data);
			} else {
				yesBranch.addData(data);
			}
		}
		if(noBranch.getStats().getTotalData() == 0 || yesBranch.getStats().getTotalData() == 0) {
			node.setLeaf();
			node.setSplitFeatureIndex(featureIndex);
			return null;
		}
		double entropyNoBranch = isPureNode(noBranch.getStats()) ? 0 : EntropyMeasure.calculateRandomness(noBranch);
		double entropyYesBranch = isPureNode(yesBranch.getStats()) ? 0 : EntropyMeasure.calculateRandomness(yesBranch);
		double combinedEntrypy = (entropyNoBranch * ((double)noBranch.getStats().getTotalData()/node.getStats().getTotalData()))+ 
				(entropyYesBranch * ((double)yesBranch.getStats().getTotalData()/node.getStats().getTotalData()));
		double infoGain = oldEntropy - combinedEntrypy;
		result.setInfoGain(infoGain);
		result.setYesBranch(yesBranch);
		result.setNoBranch(noBranch);
		result.setSplitFeatureIndex(featureIndex);
		return result;
	}

	/* (non-Javadoc)
	 * @see com.ml.hw1.classifier.DecisionClassifier#classifyData(com.ml.hw1.data.Data)
	 */
	@Override
	public double predict(TreeNode node, Data data) throws Exception {
		if(node.isLeaf()) {
			return node.getLeafValue();
		}
		if(data.getFeature(node.getSplitFeatureIndex()).isNominal()) {
			if(data.getFeatureValue(node.getSplitFeatureIndex())==0) {
				return predict(node.getNoBranch(), data);
			} else {
				return predict(node.getYesBranch(), data);
			}
		} else {
			if(data.getFeatureValue(node.getSplitFeatureIndex()) <= node.getSplitFeatureValue()) {
				return predict(node.getNoBranch(), data);
			} else {
				return predict(node.getYesBranch(), data);
			}
		}
	}
	
	private SplitNodeResult handleNumericalFeature(int featureIndex, TreeNode node) throws Exception {
		sort(node.getDataSet(), featureIndex);
		
		List<Data> dataset = node.getDataSet().getData();
		double oldEntropy = EntropyMeasure.calculateRandomness(node);
		NumericalFeatureThreshold bestThreshold = findBestThreshold(node, oldEntropy, featureIndex);
		
		if(!bestThreshold.isThresholdExist()) {
			return null;
		}
		
		TreeNode noBranch = new  TreeNode(node.depth()+1, node.getDataSet().getFeatures(), 
				node.getDataSet().getLabelIndex(), node.getDataSet().classNum());
		TreeNode yesBranch = new  TreeNode(node.depth()+1, node.getDataSet().getFeatures(), 
				node.getDataSet().getLabelIndex(), node.getDataSet().classNum());
		
		for(Data data : dataset) {
			if(data.getFeatureValue(featureIndex) <= bestThreshold.getThreshold()) {
				noBranch.addData(data);
			} else {
				yesBranch.addData(data);
			}
		}
		
		if(noBranch.getStats().getTotalData() == 0 || yesBranch.getStats().getTotalData() == 0) {
			node.setLeaf();
			node.setSplitFeatureIndex(featureIndex);
			return null;
		}
		 
		SplitNodeResult result = new SplitNodeResult();
		result.setInfoGain(bestThreshold.getInfoGain());
		result.setNoBranch(noBranch);
		result.setYesBranch(yesBranch);
		result.setSplitFeatureIndex(featureIndex);
		result.setSplitFeatureValue(bestThreshold.getThreshold());
		return result;
		
	}

	private NumericalFeatureThreshold findBestThreshold(TreeNode node, double oldEntropy, int featureIndex) throws Exception {
		double maxInfoGain = Double.NEGATIVE_INFINITY;;
		NumericalFeatureThreshold bestThreshold = null;
		NodeStats stats = node.getStats();
		double noOfOneOnLeft = 0;
		double threshold = 0;
		boolean thresholdExist = false;
		List<Data> dataset = node.getDataSet().getData();
		for(int i=0; i< dataset.size()-1; i++) {
			Data data1 = dataset.get(i);
			Data data2 = dataset.get(i+1);
			noOfOneOnLeft+= data1.labelValue();
			if(data1.getFeatureValue(featureIndex) != data2.getFeatureValue(featureIndex)) {
				threshold = ((double) data1.getFeatureValue(featureIndex) + data2.getFeatureValue(featureIndex))/2;
				NumericalFeatureThreshold nThreshold = createNumericalFeatureThreshold(i, noOfOneOnLeft, node, stats, threshold, 
						oldEntropy, node.getDataSet().isClassificationTask());
				double infoGain = nThreshold.getInfoGain();
				if(infoGain > maxInfoGain) {
					maxInfoGain = infoGain;
					bestThreshold = nThreshold;
					thresholdExist = true;
				}
			}
		}
		if(!thresholdExist) {
			return new  NumericalFeatureThreshold();
		}
		return bestThreshold;
	}

	private NumericalFeatureThreshold createNumericalFeatureThreshold(int rowIndex, double leftSum, TreeNode node, NodeStats stats, 
			double threshold, double oldEntropy, boolean isClassificationTask) throws Exception {
		return new NumericalFeatureThreshold(stats, rowIndex, leftSum, node, threshold, oldEntropy, isClassificationTask);
	}

	private void sort(DataSet dataSet, final int featureIndex) {
		if(dataSet != null) {
			Collections.sort(dataSet.getData(), new Comparator<Data>() {

				@Override
				public int compare(Data arg0, Data arg1) {
					Double value1 = new Double(arg0.getFeatureValue(featureIndex));
					Double value2 = new Double(arg1.getFeatureValue(featureIndex));
					return value1.compareTo(value2);
				}
			});
		}
	}
}
