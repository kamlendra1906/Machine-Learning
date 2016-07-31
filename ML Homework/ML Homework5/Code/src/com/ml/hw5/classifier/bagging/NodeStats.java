/**
 * 
 */
package com.ml.hw5.classifier.bagging;

import com.ml.hw5.data.Data;

/**
 * @author Kamlendra Kumar
 *
 */
public class NodeStats {

	private int[] labelsPerClass; // Used for categorical data
	private int totalData;
	private double totalValue;
	private double sumOfLabelSquared;
	
	public NodeStats() {
		totalData = 0;
		totalValue = 0;
		sumOfLabelSquared = 0;
	}
	
	public NodeStats(int classNum) {
		labelsPerClass = new int[classNum];
		totalData = 0;
		totalValue = 0;
		sumOfLabelSquared = 0;
	}
	
	
	public void add(Data data) throws Exception {
		int labelIndex = data.labelIndex();
		if(data.getFeature(labelIndex).isNominal()) {
			labelsPerClass[(int) data.labelValue()]++;
		} else {
			totalValue += data.labelValue();
			sumOfLabelSquared += Math.pow(data.labelValue(), 2);
		}
		totalData++;
	}
	
	
	public int predictClass() {
		int predictedClass = -1;
		int maxLabelCount = -1;
		for(int i=0; i< labelsPerClass.length; i++) {
			if(labelsPerClass[i] > maxLabelCount) {
				predictedClass = i;
				maxLabelCount = labelsPerClass[i];
			}
		}
		return predictedClass;
	}
	
	public double getMean() {
		return totalValue/totalData;
	}
	
	public int[] getLabelPerClass() {
		return labelsPerClass;
	}
	
	public int getTotalData() {
		return totalData;
	}

	/**
	 * @return the totalValue
	 */
	public double getTotalValue() {
		return totalValue;
	}

	/**
	 * @param totalValue the totalValue to set
	 */
	public void setTotalValue(double totalValue) {
		this.totalValue = totalValue;
	}

	/**
	 * @return the sumOfLabelSquared
	 */
	public double getSumOfLabelSquared() {
		return sumOfLabelSquared;
	}

	/**
	 * @param sumOfLabelSquared the sumOfLabelSquared to set
	 */
	public void setSumOfLabelSquared(double sumOfLabelSquared) {
		this.sumOfLabelSquared = sumOfLabelSquared;
	}
}
