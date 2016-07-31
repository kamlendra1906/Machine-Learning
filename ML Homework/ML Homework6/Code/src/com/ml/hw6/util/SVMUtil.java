package com.ml.hw6.util;

import com.ml.hw6.data.Data;
import com.ml.hw6.data.DataSet;
import com.ml.hw6.data.SVMData;

import libsvm.svm_node;

/**
 * @author kkumar
 *
 */
public class SVMUtil {
	
	public static final String SVM_PARAMETER_C = "c";
	public static final String SVM_PARAMETER_TOLERANCE = "t";
	public static final String SVM_PARAMETER_DEGREE = "d";
	public static final String SVM_PARAMETER_ALPHA = "alpha";
	public static final String SVM_PARAMETER_BETA = "beta";
	public static final String SVM_PARAMETER_NU = "nu";
	public static final String SVM_PARAMETER_PASSES = "passes";
	public static final String SVM_PARAMETERS_KERNEL_CACHE_ENABLED = "kernelCacheEnabled";
	public static final String SSVM_PARAMETERS_FX_CACHE_ENABLED = "fxCacheEnabled";
	public static final String SVM_PARAMETER_KERNEL_TYPE = "kernel_type";
	public static final int LINEAR_KERNEL = 0;
	public static final int POLY_KERNEL = 1;
	public static final int RBF_KERNEL = 2;

	public static SVMData getSVMData(DataSet dataset) throws Exception {
		int featureSize = dataset.getFeatures().size() -1;
		int dataSize = dataset.dataSize();
		
		SVMData svmDataset = new SVMData(dataSize, featureSize);
		double[] labels = svmDataset.getLabels();
		svm_node[][] svmData = svmDataset.getData();
		
		for(int i=0; i < dataSize; i++) {
			Data data = dataset.getData().get(i);
			svmData[i] = getSVMNodes(data);
			labels[i] = data.labelValue();
		}
		return svmDataset;
	}

	private static svm_node[] getSVMNodes(Data data) throws Exception {
		int featureSize = data.getFeatures().size() -1;
		svm_node[] nodes = new svm_node[featureSize];
		
		for(int i=0; i<featureSize; i++) {
			svm_node node = new svm_node();
			node.index = i;
			node.value = data.getFeatureValue(i);
			nodes[i] = node;
		}
		return nodes;
	}
}