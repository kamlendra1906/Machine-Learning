/**
 * 
 */
package com.ml.hw7.stats;

import com.ml.hw7.data.Data;

/**
 * @author kkumar
 *
 */
public class ActiveLearningDataStats implements Comparable<ActiveLearningDataStats>{

	private Data data;
	private double predictedLabel;
	private double distanceFromThreshold;
	
	@Override
	public int compareTo(ActiveLearningDataStats o) {
		if(this.distanceFromThreshold == o.getDistanceFromThreshold()) {
			return 0;
		}
		if(this.distanceFromThreshold > o.getDistanceFromThreshold()) {
			return 1;
		}
		return -1;
	}

	/**
	 * @return the data
	 */
	public Data getData() {
		return data;
	}

	/**
	 * @param data the data to set
	 */
	public void setData(Data data) {
		this.data = data;
	}

	/**
	 * @return the predictedLabel
	 */
	public double getPredictedLabel() {
		return predictedLabel;
	}

	/**
	 * @param predictedLabel the predictedLabel to set
	 */
	public void setPredictedLabel(double predictedLabel) {
		this.predictedLabel = predictedLabel;
	}

	/**
	 * @return the distanceFromThreshold
	 */
	public double getDistanceFromThreshold() {
		return distanceFromThreshold;
	}

	/**
	 * @param distanceFromThreshold the distanceFromThreshold to set
	 */
	public void setDistanceFromThreshold(double distanceFromThreshold) {
		this.distanceFromThreshold = distanceFromThreshold;
	}
}
