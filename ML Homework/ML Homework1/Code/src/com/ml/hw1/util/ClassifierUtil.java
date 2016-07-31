/**
 * 
 */
package com.ml.hw1.util;

import java.util.List;

import com.ml.hw1.data.Data;
import com.ml.hw1.data.DataForRegression;
import com.ml.hw1.data.DataSet;

/**
 * @author kkumar
 *
 */
public class ClassifierUtil {


	public static double logValue(double probability) {
		return Math.log(probability)/Math.log(2);
	}
	
	public static DataForRegression prepareData(DataSet dataSet) throws Exception {
		if (dataSet != null && dataSet.dataSize() > 0) {
			List<Data> data = dataSet.getData();
			DataForRegression dataForRegression = new DataForRegression();
			double[] featureMatrix = null;
			double[] valueMatrix = null;
			int sampleSize = data.size();
			int featureSize = dataSet.getFeatures().size();
			
			int size = sampleSize * featureSize;
			featureMatrix = new double[size];
			double[][] twoDFeatureMatrix = new double[sampleSize][featureSize];
			valueMatrix = new double[data.size()];
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < featureSize; j++) {
					if(j ==0) {
						featureMatrix[featureSize * i + j] = 1;
						twoDFeatureMatrix[i][j] = 1;
					} else {
						featureMatrix[featureSize * i + j] = data.get(i).getFeatureValue(j-1);
						twoDFeatureMatrix[i][j] = data.get(i).getFeatureValue(j-1);
					}
				}
				valueMatrix[i] = data.get(i).labelValue();
			}
			dataForRegression.setFeatureData(featureMatrix);
			dataForRegression.setTwoDArrayFeatureData(twoDFeatureMatrix);
			dataForRegression.setValueData(valueMatrix);
			dataForRegression.setFeatureSize(featureSize);
			dataForRegression.setSampleSize(sampleSize);
			return dataForRegression;
		}
		return null;
	}
}
