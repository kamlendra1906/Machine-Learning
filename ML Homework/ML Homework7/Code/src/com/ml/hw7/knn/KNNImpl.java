package com.ml.hw7.knn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;

public class KNNImpl {

	private int k;
	private double range;
	private boolean useK;
	private SVMKernel kernal;
	private Comparator<KNNNeighbour> ranker;
	
	
	public KNNImpl(int k, SVMKernel kernel, double range, boolean useK, Comparator<KNNNeighbour> ranker) {
		this.k = k;
		this.kernal = kernel;
		this.useK = useK;
		this.range = range;
		this.ranker = ranker;
	}

	public double runKNN(DataSet trainingData, DataSet testData, Map<String, Object> additionalData) throws Exception {
		double error = 0;
		for(Data testPoint : testData.getData()) {
			double actualLabel = testPoint.labelValue();
			double predictedLabel = predictLabel(trainingData, testPoint, additionalData);
			if(Double.isNaN(predictedLabel)) {
				continue;
			}
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/testData.dataSize();
	}

	private double predictLabel(DataSet trainingData, Data testPoint, Map<String, Object> additionalData) throws Exception {
		List<KNNNeighbour> neighbours = getClosestNeighbours(trainingData.getData(), testPoint, additionalData);
		if(neighbours.size() == 0) {
			return Double.NaN;
		}
		return getVotedLabel(neighbours);
		
	}
	
	private List<KNNNeighbour> getClosestNeighbours(List<Data> trainingData, Data testPoint, Map<String, Object> additionalData) throws Exception {
		List<KNNNeighbour> allNeighbours = getNeighbours(trainingData, testPoint, additionalData);
		if(useK) {
			return getKNearestNeighbours(allNeighbours, k);
		}
		return getNeighboursWithinRange(allNeighbours, range);
	}

	private double getVotedLabel(List<KNNNeighbour> neighbours) throws Exception {
		Map<Double, Integer> labelNeighboursCount = new HashMap<Double, Integer>();
		for(KNNNeighbour neighbour : neighbours) {
			Data data = neighbour.getPoint();
			if(labelNeighboursCount.containsKey(data.labelValue())) {
				labelNeighboursCount.put(data.labelValue(), (int)labelNeighboursCount.get(data.labelValue())+1);
			} else {
				labelNeighboursCount.put(data.labelValue(), 1);
			}
		}
		
		double maxLabelCount = Double.NEGATIVE_INFINITY;
		double maxLabel = Double.NaN;
		
		for(Entry<Double, Integer> entry : labelNeighboursCount.entrySet()) {
			double label = entry.getKey();
			int labelCount = entry.getValue();
			if(labelCount > maxLabelCount) {
				maxLabelCount = labelCount;
				maxLabel = label;
			}
		}
		return maxLabel;
	}
	
	private List<KNNNeighbour> getNeighboursWithinRange(List<KNNNeighbour> allNeighbours, double range) {
		List<KNNNeighbour> neighboursInRange = new ArrayList<KNNNeighbour>();
		for(KNNNeighbour neighbour : allNeighbours) {
			if(neighbour.getSimilarity() <= range) {
				neighboursInRange.add(neighbour);
			}
		}
		return neighboursInRange;
	}

	private List<KNNNeighbour> getKNearestNeighbours(List<KNNNeighbour> allNeighbours, int k) {
		List<KNNNeighbour> nearestNeighbours = new ArrayList<KNNNeighbour>();
		for(int i=0; i < k; i++) {
			nearestNeighbours.add(allNeighbours.get(i));
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
