/**
 * 
 */
package com.ml.hw5.data;

import java.util.List;

import com.ml.hw5.feature.Feature;

/**
 * @author Kamlendra Kumar
 *
 */
public class Data {

	private double[] featureValues;
	private DataSet dataset;
	
	/**
	 * Constructor
	 * @param data
	 */
	public Data(Data data) {
		featureValues = data.getFeatureValues();
		dataset = data.getDataSet();
	}
	
	public Data(int featureSize) {
		featureValues = new double[featureSize];
	}
	
	/**
	 * Constructor
	 * @param values
	 */
	public Data(double[] values) {
		this.featureValues = values;
		this.dataset = null;
	}
	
	/**
	 * Returns the feature values for this data
	 * @return
	 */
	public double[] getFeatureValues() {
		return featureValues;
	}
	
	/**
	 * Returns the value of a given feature for the data.
	 * @param index
	 * @return
	 */
	public double getFeatureValue(int index) {
		return featureValues[index];
	}
	
	/**
	 * Returns the dataset this data is associated with
	 * @return
	 */
	public DataSet getDataSet() {
		return dataset;
	}
	
	/**
	 * Assigns the Dataset to this data.
	 * @param dataset
	 */
	public void setDataSet(DataSet dataset) {
		this.dataset = dataset;
	}
	
	/**
	 * Returns the index of class label.
	 * @return
	 * @throws Exception
	 */
	public int labelIndex() throws Exception {
		if(dataset == null) {
			throw new Exception("DataSet is null");
		}
		return dataset.getLabelIndex();
	}
	
	/**
	 * Returns the class label.
	 * @return
	 * @throws Exception
	 */
	public double labelValue() throws Exception {
		return featureValues[labelIndex()];
	}
	
	public void setLabelValue(double value) throws Exception {
		this.featureValues[labelIndex()] = value;
	}
	
	/**
	 * Returns the feature at a given index.
	 * @param index
	 * @return
	 * @throws Exception if feature doesn't exist.
	 */
	public Feature getFeature(int index) throws Exception {
		if(dataset == null) {
			throw new Exception("DataSet is null");
		}
		return dataset.getFeature(index);
	}
	
	/**
	 * Returns the features of this data.
	 * @return
	 * @throws Exception
	 */
	public List<Feature> getFeatures() throws Exception {
		if(dataset == null) {
			throw new Exception("Dataset is null");
		}
		return dataset.getFeatures();
	}
	
	/**
	 * 
	 * @param labelValueIndex
	 * @param value
	 */
	public void setFeatureValue(int labelValueIndex, double value) {
		this.featureValues[labelValueIndex] = value;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new  StringBuilder();
		for(double value : featureValues) {
			builder.append(String.valueOf(value)+" ");
		}
		return builder.toString();
	}
}