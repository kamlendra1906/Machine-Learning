/**
 * 
 */
package src.com.ml.hw3.data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import src.com.ml.hw3.feature.Feature;

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
	
	private double[] featureDataAverage;
	private double[] featureMin;
	private double[] featureMax;
	
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
		featureDataAverage = new double[features.size() -1];
		featureMin = initializeFeatureMinOrMax(features.size() -1, true);
		featureMax = initializeFeatureMinOrMax(features.size() -1, false);
	}
	
	
	
	private double[] initializeFeatureMinOrMax(int featureSize, boolean featureMin) {
		double featureValue = featureMin ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		double[] feature = new double[featureSize];
		for(int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
			feature[featureIndex] = featureValue;
		}
		return feature;
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
		reCalculateDataFeatureAverage(data);
		reCalculateFeatureMinAndMax(data);
	}
	
	private void reCalculateFeatureMinAndMax(Data data) {
		int featureSize = this.features.size() -1;
		for(int feature = 0; feature < featureSize; feature++) {
			double featureValue = data.getFeatureValue(feature);
			if(featureValue > this.featureMax[feature]) {
				this.featureMax[feature] = featureValue;
			}
			if(featureValue < this.featureMin[feature]) {
				this.featureMin[feature] = featureValue;
			}
		}
	}

	private void reCalculateDataFeatureAverage(Data data) {
		int featureSize = this.features.size() -1;
		for(int feature = 0; feature <  featureSize; feature++) {
			featureDataAverage[feature] += data.getFeatureValue(feature);
		}
	}

	public double[] getDataFeatureMean() throws Exception {
		int featureSize = this.features.size() -1;
		double[] featureMean = new double[featureSize];
		for(int feature = 0; feature <  featureSize; feature++) {
			featureMean[feature] = featureDataAverage[feature] / dataSize();
		}
		return featureMean;
	}
	
	public double[][] getFeatureDataAsArray() throws Exception {
		int featureSize = this.features.size() -1;
		double[][] featureDataArray = new double[dataSize()][featureSize];
		
		for(int row=0; row < this.dataSize(); row++) {
			Data dataPoint = this.data.get(row);
			for(int col=0; col < featureSize; col++) {
				featureDataArray[row][col] = dataPoint.getFeatureValue(col);
			}
		}
		return featureDataArray;
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
	
	/**
	 * @return the featureDataAverage
	 */
	public double[] getFeatureDataAverage() {
		return featureDataAverage;
	}

	/**
	 * @return the featureMin
	 */
	public double[] getFeatureMin() {
		return featureMin;
	}



	/**
	 * @return the featureMax
	 */
	public double[] getFeatureMax() {
		return featureMax;
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
