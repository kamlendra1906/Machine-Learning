/**
 * 
 */
package com.ml.hw7.knn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;

/**
 * @author kkumar
 *
 */
public class ReliefImpl {

	private int k;
	private SVMKernel kernal;
	private Comparator<KNNNeighbour> ranker;
	private double[] featureWeights;
	
	
	public ReliefImpl(int k, SVMKernel kernel, int featureSize, Comparator<KNNNeighbour> ranker) {
		this.k = k;
		this.kernal = kernel;
		this.ranker = ranker;
		this.featureWeights = new double[featureSize];
	}

	public List<Integer> runRelief(DataSet trainingData, Map<String, Object> additionalData, int featureNum) throws Exception {
		List<Data> trainingPoints = trainingData.getData();
		for(Data data : trainingPoints) {
			double label = data.labelValue();
			List<KNNNeighbour> nearestNeighbours = getClosestNeighbours(trainingPoints, data, additionalData);
			
			List<Data> samePoints = new ArrayList<Data>();
			List<Data> diffPoints = new ArrayList<Data>();
			
			for(KNNNeighbour neighbour : nearestNeighbours) {
				if(neighbour.getPoint().labelValue() == label) {
					samePoints.add(neighbour.getPoint());
				} else {
					diffPoints.add(neighbour.getPoint());
				}
			}
			
			updateFeatureQuality(samePoints, diffPoints, data);
		}
		
		List<FeatureRank> featureRank = new ArrayList<FeatureRank>();
		for(int i=0; i < featureWeights.length; i++) {
			featureRank.add(new FeatureRank(i, featureWeights[i]));
		}
		Collections.sort(featureRank);
		
		List<Integer> features = new ArrayList<Integer>();
		for(int i=0; i < featureNum; i++) {
			features.add(featureRank.get(i).getFeatureId());
		}
		return features;
	}

	private void updateFeatureQuality(List<Data> samePoints, List<Data> diffPoints, Data data) {
		for(Data samePoint : samePoints) {
			for(int i=0; i < featureWeights.length; i++) {
				featureWeights[i]-= Math.pow((data.getFeatureValue(i) - samePoint.getFeatureValue(i)), 2);
			}
		}
		for(Data diffPoint : diffPoints) {
			for(int i=0; i < featureWeights.length; i++) {
				featureWeights[i]+= Math.pow(data.getFeatureValue(i) - diffPoint.getFeatureValue(i), 2);
			}
		}
	}

	private List<KNNNeighbour> getClosestNeighbours(List<Data> trainingData, Data testPoint, Map<String, Object> additionalData) throws Exception {
		List<KNNNeighbour> allNeighbours = getNeighbours(trainingData, testPoint, additionalData);
		return getKNearestNeighbours(allNeighbours, testPoint);
	}

	private List<KNNNeighbour> getKNearestNeighbours(List<KNNNeighbour> allNeighbours, Data dataPoint) {
		List<KNNNeighbour> nearestNeighbours = new ArrayList<KNNNeighbour>();
		int i=0;
		while(nearestNeighbours.size() != k) {
			KNNNeighbour neighbour = allNeighbours.get(i++);
			if(neighbour.getPoint().getDataId() != dataPoint.getDataId()) {
				nearestNeighbours.add(neighbour);
			}
		}
		return nearestNeighbours;
	}

	private List<KNNNeighbour> getNeighbours(List<Data> trainingData, Data testPoint, Map<String, Object> additionalData) throws Exception {
		List<KNNNeighbour> neighbours = new ArrayList<KNNNeighbour>();
		for(Data trainingPoint : trainingData) {
			neighbours.add(new KNNNeighbour(kernal.evaluateKernel(trainingPoint, testPoint, 0, 0, additionalData), trainingPoint));
		}
		Collections.sort(neighbours, ranker);
		return neighbours;
	}
}

class FeatureRank implements Comparable<FeatureRank> {
	private int featureId;
	private double quality;
	
	public FeatureRank(int featureId, double quality) {
		this.featureId = featureId;
		this.quality = quality;
	}

	
	@Override
	public int compareTo(FeatureRank o) {
		if(this.quality == o.getQuality()) {
			return 0;
		}
		if(this.quality > o.getQuality()) {
			return -1;
		}
		return 1;
	}


	/**
	 * @return the featureId
	 */
	public int getFeatureId() {
		return featureId;
	}


	/**
	 * @param featureId the featureId to set
	 */
	public void setFeatureId(int featureId) {
		this.featureId = featureId;
	}


	/**
	 * @return the quality
	 */
	public double getQuality() {
		return quality;
	}


	/**
	 * @param quality the quality to set
	 */
	public void setQuality(double quality) {
		this.quality = quality;
	}
}
