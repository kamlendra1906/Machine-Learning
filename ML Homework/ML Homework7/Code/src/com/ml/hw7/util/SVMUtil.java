package com.ml.hw7.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.ml.hw7.data.Data;
import com.ml.hw7.data.DataSet;
import com.ml.hw7.data.SVMData;

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
	public static final int DOT_PRODUCT_SIMILARITY = 0;
	public static final int RBF_SIMILARITY = 1;
	public static final int DISTANCE_SIMILARITY = 2;
	public static final int COSINE_SIMILARITY = 3;
	public static final String USE_WEIGHTED_NEIGHBOURS = "weightedNeighbours";

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
	
	public static double calculateDotProduct(Data x1, Data x2) throws Exception {
		double dotProduct = 0;
		int featureSize = x1.getFeatures().size();
		int labelIndex = x1.labelIndex();
		for(int i=0; i < featureSize; i++) {
			if(i != labelIndex) {
				dotProduct+= x1.getFeatureValue(i) * x2.getFeatureValue(i);
			}
		}
		return dotProduct;
	}
	
	public static double calculateNormOfData(Data x1) throws Exception {
		double norm = 0;
		int featureSize = x1.getFeatures().size();
		int labelIndex = x1.labelIndex();
		for(int i=0; i < featureSize; i++) {
			if(i != labelIndex) {
				norm+= Math.pow(x1.getFeatureValue(i),2);
			}
		}
		return Math.sqrt(norm);
	}
	
	public static List<Data> sampleTrainingData(List<Data> trainingData, double samplePercent) throws Exception {
		List<Data> result = new ArrayList<Data>();
		@SuppressWarnings("unchecked")
		List<Data>[] imageListByClass = new ArrayList[10];
		
		for(Data img : trainingData) {
			List<Data> list = imageListByClass[(int)img.labelValue()];
			if(list == null) {
				list = new ArrayList<Data>();
				imageListByClass[(int)img.labelValue()] = list;
			}
			list.add(img);
		}
		
		for(List<Data> images : imageListByClass) {
			Collections.shuffle(images);
			sample(images, result, samplePercent);
		}
		return result;
	}
	
	private static void sample(List<Data> images, List<Data> result, double samplePercent) {
		int numOfDataToBeAdded = (int) ((images.size() * samplePercent) /100);
		for(int i=0; i < numOfDataToBeAdded; i++) {
			result.add(images.get(i));
		}
	}
  }