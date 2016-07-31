package com.ml.hw7.knn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.kami.hw7.svm.kernel.SVMKernel;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.KNNNeighbour;

public class ParzenWindowsImpl {

	private SVMKernel kernal;
	
	public ParzenWindowsImpl(SVMKernel kernel) {
		this.kernal = kernel;
	}

	public double runKNN(DataSet trainingData, DataSet testData, Map<String, Object> additionalData) throws Exception {
		double error = 0;
		for(Data testPoint : testData.getData()) {
			double actualLabel = testPoint.labelValue();
			double predictedLabel = predictLabel(trainingData, testPoint, additionalData);
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/testData.dataSize();
	}

	private double predictLabel(DataSet trainingData, Data testPoint, Map<String, Object> additionalData) throws Exception {
		List<KNNNeighbour> neighbours = getNeighbours(trainingData.getData(), testPoint, additionalData);
		return getVotedLabel(neighbours);
		
	}
	
	
	private double getVotedLabel(List<KNNNeighbour> neighbours) throws Exception {
		Map<Double, Double> labelNeighboursCount = new HashMap<Double, Double>();
		for(KNNNeighbour neighbour : neighbours) {
			Data data = neighbour.getPoint();
			if(labelNeighboursCount.containsKey(data.labelValue())) {
				labelNeighboursCount.put(data.labelValue(), labelNeighboursCount.get(data.labelValue()) + neighbour.getSimilarity());
			} else {
				labelNeighboursCount.put(data.labelValue(), neighbour.getSimilarity());
			}
		}
		
		double maxLabelContribution = Double.NEGATIVE_INFINITY;
		double maxLabel = Double.NaN;
		
		for(Entry<Double, Double> entry : labelNeighboursCount.entrySet()) {
			double label = entry.getKey();
			double labelCount = entry.getValue();
			if(labelCount > maxLabelContribution) {
				maxLabelContribution = labelCount;
				maxLabel = label;
			}
		}
		return maxLabel;
	}
	
	private List<KNNNeighbour> getNeighbours(List<Data> trainingData, Data testPoint, Map<String, Object> additionalData) throws Exception {
		List<KNNNeighbour> neighbours = new ArrayList<KNNNeighbour>();
		for(Data trainingPoint : trainingData) {
			neighbours.add(new KNNNeighbour(kernal.evaluateKernel(trainingPoint, testPoint, 0, 0, additionalData), trainingPoint));
		}
		return neighbours;
	}
}