/**
 * 
 */
package com.ml.hw6.data;

import libsvm.svm_node;

/**
 * @author kkumar
 *
 */
public class SVMData {

	private double[] labels;
	private svm_node[][] data;
	
	public SVMData(int dataSize, int featureSize) {
		this.labels = new double[dataSize];
		this.data = new svm_node[dataSize][featureSize];
	}

	/**
	 * @return the labels
	 */
	public double[] getLabels() {
		return labels;
	}

	/**
	 * @param labels the labels to set
	 */
	public void setLabels(double[] labels) {
		this.labels = labels;
	}

	/**
	 * @return the data
	 */
	public svm_node[][] getData() {
		return data;
	}

	/**
	 * @param data the data to set
	 */
	public void setData(svm_node[][] data) {
		this.data = data;
	}
}
