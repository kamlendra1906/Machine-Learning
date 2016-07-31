/**
 * 
 */
package com.ml.hw2.data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import com.ml.hw2.feature.Feature;

/**
 * @author Kamlendra Kumar
 *
 */
public class DataSet {

	private List<Data> data;
	private List<Feature> features;
	private int labelIndex;
	private HashSet<Integer> classes;
	private int classNum;
	
	/**
	 * Constructor
	 * @param labelIndex
	 * @param features
	 * @param data
	 */
	public DataSet(int labelIndex, List<Feature> features) {
		this.labelIndex = labelIndex;
		this.features = features;
		data = new ArrayList<Data>();
		classes = new HashSet<Integer>();
	}
	
	/**
	 * Returns the size of the dataset.
	 * @return
	 * @throws Exception
	 */
	public int dataSize() throws Exception {
		if(data == null) {
			throw new Exception("data is null");
		}
		return data.size();
	}
	
	/**
	 * Returns the features of the dataset.
	 * @return
	 * @throws Exception
	 */
	public List<Feature> getFeatures() throws Exception {
		if(features == null || features.size() == 0) {
			throw new Exception("Feature is null or empty");
		}
		return features;
	}
	
	/**
	 * Returns the feature at given index.
	 * @param index
	 * @return
	 * @throws Exception
	 */
	public Feature getFeature(int index) throws Exception {
		if(features == null || features.size() == 0) {
			throw new Exception("No features found");
		}
		if( (index < 0) || (index > (features.size() -1))) {
			throw new Exception("Feature index out of bound");
		}
		return features.get(index);
	}
	
	/**
	 * Adds data to the dataset.
	 * @param data
	 * @throws Exception 
	 */
	public void addData(Data data) throws Exception {
		data.setDataSet(this);
		this.data.add(data);
		if(features.get(labelIndex).isNominal()) {
			Integer classLabel = new Integer((int) data.labelValue());
			classes.add(classLabel);
			classNum = classes.size();
		}
	}
	
	/**
	 * Returns the index of data label
	 * @return
	 */
	public int getLabelIndex() {
		return this.labelIndex;
	}
	
	public void setClassNum(int classNum) {
		this.classNum = classNum;
	}
	
	public int classNum() {
		return classNum;
	}
	
	public boolean isClassificationTask() {
		return features.get(labelIndex).isNominal();
	}
	
	public List<Data> getData() {
		return data;
	}
	
	@Override
	public String toString() {
		StringBuilder stringBuilder = new StringBuilder();
		for(Data d : data) {
			stringBuilder.append(d.toString()+"\n");
		}
		return stringBuilder.toString();
	}
}
