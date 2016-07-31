/**
 * 
 */
package com.ml.hw6.classifier.bagging;

import java.util.List;

import com.ml.hw6.data.Data;

/**
 * @author Kamlendra Kumar
 *
 */
public class NumericalFeatureThreshold {

	private double threshold;
	private double noOfOnesOnLeft;
	private int noOfDataOnLeft;
	private NodeStats stats;
	private double infoGain;
	private boolean thresholdExist;
	
	
	public NumericalFeatureThreshold(NodeStats stats, int rowIndex, double leftSum, TreeNode node, double threshold, 
			double oldEntropy, boolean isClassificationTask) throws Exception {
		this.threshold = threshold;
		this.noOfOnesOnLeft = leftSum;
		this.noOfDataOnLeft = rowIndex+1;
		this.stats = stats;
		this.infoGain = getInfoGain(oldEntropy, isClassificationTask, node);
		thresholdExist = true;
	}

	public NumericalFeatureThreshold() {
		thresholdExist = false;
	}

	private double getInfoGain(double oldEntropy, boolean isClassificationTask, TreeNode node) throws Exception {
		if(isClassificationTask) {
			return getInfoGainForClassification(oldEntropy);
		}
		return getInfoGainForRegression(oldEntropy, node);
	}
	
	private double getInfoGainForRegression(double oldEntropy, TreeNode node) throws Exception {
		double totalValueOfNode = stats.getTotalValue();
		double totalValueOnRight = totalValueOfNode - noOfOnesOnLeft;
		int noOfDataOnRight = this.stats.getTotalData() - noOfDataOnLeft;
		double meanLeft = noOfOnesOnLeft/noOfDataOnLeft;
		double meanRight = totalValueOnRight/noOfDataOnRight;
		double leftMSE = 0;
		double rightMSE = 0;
		List<Data> datas = node.getDataSet().getData();
		int counter = 0;
		for(Data data : datas) {
			if(counter < noOfDataOnLeft) {
				leftMSE+= Math.pow(meanLeft - data.labelValue(), 2);
			} else {
				rightMSE+= Math.pow(meanRight - data.labelValue(), 2);
			}
			counter++;
		}
		leftMSE/= noOfDataOnLeft;
		rightMSE/= noOfDataOnRight;
		double combinedReduction = leftMSE * noOfDataOnLeft/stats.getTotalData() + rightMSE * noOfDataOnRight/stats.getTotalData();
		//return oldEntropy - (leftMSE+rightMSE);
		return oldEntropy - combinedReduction;
	}

	private double getInfoGainForClassification(double oldEntropy) {
		double totalNumberOfOnes = this.stats.getLabelPerClass()[1];
		double noOfOnesOnRight = totalNumberOfOnes - noOfOnesOnLeft;
		double noOfDataOnRight = this.stats.getTotalData() - noOfDataOnLeft;
		double noOfZeroOnLeft = noOfDataOnLeft - noOfOnesOnLeft;
		double noOfZeroOnRight = noOfDataOnRight - noOfOnesOnRight;
		
		double leftEntropy = getEntropy(noOfZeroOnLeft, noOfOnesOnLeft, noOfDataOnLeft);
		double rightEntropy = getEntropy(noOfZeroOnRight, noOfOnesOnRight, noOfDataOnRight);
		double combinedEntropy = ((double)noOfDataOnLeft)/stats.getTotalData()* leftEntropy +
				((double)noOfDataOnRight)/stats.getTotalData()* rightEntropy;
		return oldEntropy - combinedEntropy;
		
	}
	
	private double getEntropy(double noOfZero, double noOfOne, double totalData) {
		if(noOfZero == 0 || noOfOne == 0) {
			return 0;
		}
		double probability1 = noOfZero/totalData; 
		double probability2 = noOfOne/totalData;
		double entropy = probability1*Math.log(probability1)/Math.log(2) + probability2*Math.log(probability2)/Math.log(2);
		return -entropy;
	}
	
	
	public double getThreshold() {
		return threshold;
	}


	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}


	public double getInfoGain() {
		return infoGain;
	}


	public void setInfoGain(double infoGain) {
		this.infoGain = infoGain;
	}


	public boolean isThresholdExist() {
		return thresholdExist;
	}


	public void setThresholdExist(boolean thresholdExist) {
		this.thresholdExist = thresholdExist;
	}
}