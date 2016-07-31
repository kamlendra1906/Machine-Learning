package com.ml.hw5.classifier.bagging;

import com.ml.hw5.data.Data;
import com.ml.hw5.util.ClassifierUtil;

public class EntropyMeasure  {

	public static double calculateRandomness(TreeNode node) throws Exception {
		NodeStats stats = node.getStats();
		int totalData = stats.getTotalData();
		if(!node.getDataSet().isClassificationTask()) {
			double nodeMean = node.getStats().getMean();
			double squaredError = 0;
			for(Data data : node.getDataSet().getData()) {
				squaredError+= Math.pow(data.labelValue() - nodeMean, 2);
			}
			return squaredError/node.getDataSet().dataSize();
		}
		int [] labelCountPerClass = stats.getLabelPerClass();
		double entropy = 0;
		for(int classs=0; classs < labelCountPerClass.length; classs++) {
			int labelCount = labelCountPerClass[classs];
			double probability = ((double)labelCount/totalData);
			double temp = probability*ClassifierUtil.logValue(probability);
			entropy+= temp;
		}
		return -entropy;
	}

}
