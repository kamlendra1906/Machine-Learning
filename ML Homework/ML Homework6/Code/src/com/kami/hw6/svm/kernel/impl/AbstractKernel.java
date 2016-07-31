/**
 * 
 */
package com.kami.hw6.svm.kernel.impl;

import java.util.HashMap;
import java.util.Map;

import com.kami.hw6.svm.kernel.SVMKernel;
import com.ml.hw6.data.Data;
import com.ml.hw6.data.SVMPoint;
import com.ml.hw6.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class AbstractKernel implements SVMKernel {
	
	private Map<SVMPoint, Double> pairSimilarityMap;
	private double[][] pairSimilarityArray;
	
	public AbstractKernel(int size) {
		pairSimilarityMap = new HashMap<SVMPoint, Double>();
		pairSimilarityArray = new double[size][size];
		for(int i=0;i<size;i++) {
			for(int j=0;j<size;j++) {
				pairSimilarityArray[i][j] = Double.NaN;
			}
		}
	}

	/* (non-Javadoc)
	 * @see com.kami.hw6.svm.kernel.SVMKernel#evaluateKernel(com.ml.hw6.data.Data, com.ml.hw6.data.Data, int, int, java.util.Map)
	 */
	/*@Override
	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData)
			throws Exception {
		Double dotProduct = null;
		boolean cacheEnabled = (boolean) additionalData.get(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED);
		if(cacheEnabled) {
			SVMPoint point = new SVMPoint(indexX1, indexX2);
			dotProduct = pairSimilarityMap.get(point);
			if(dotProduct == null) {
				dotProduct = evaluateDotProduct(x1, x2);
				pairSimilarityMap.put(point, dotProduct);
			}
		} else {
			dotProduct = evaluateDotProduct(x1, x2);
		}
		return dotProduct;
	}*/
	
	
	@Override
	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData)
			throws Exception {
		double dotProduct = Double.NaN;
		boolean cacheEnabled = (boolean) additionalData.get(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED);
		if(cacheEnabled) {
			dotProduct = pairSimilarityArray[indexX1][indexX2];
			if(Double.isNaN(dotProduct)) {
				dotProduct = evaluate(x1, x2, additionalData);
				pairSimilarityArray[indexX1][indexX2] = dotProduct;
				pairSimilarityArray[indexX2][indexX1] = dotProduct;
			}
		} else {
			dotProduct = evaluate(x1, x2, additionalData);
		}
		if(Double.isNaN(dotProduct)) {
			throw new Exception("NaN from Kernal");
		}
		return dotProduct;
	}

	protected double evaluate(Data x1, Data x2, Map<String, Object> additionalData) throws Exception {
		if(SVMUtil.RBF_KERNEL == (int) additionalData.get(SVMUtil.SVM_PARAMETER_KERNEL_TYPE)) {
			return rbfKernelEvaluation(x1, x2, additionalData);
		}
		return evaluateDotProduct(x1, x2, additionalData);
	}
	
	private double rbfKernelEvaluation(Data x1, Data x2, Map<String, Object> additionalData) throws Exception {
		double nu = (double) additionalData.get(SVMUtil.SVM_PARAMETER_NU);
		int featureSize = x1.getFeatures().size() -1;
		double vectorDiff = 0;
		for(int i=0; i < featureSize; i++) {
			vectorDiff+= Math.pow((x1.getFeatureValue(i) - x2.getFeatureValue(i)),2);
		}
		vectorDiff/= (2*nu*nu);
		return Math.exp(-vectorDiff);
	}

	protected double evaluateDotProduct(Data x1, Data x2, Map<String, Object> additionalData) throws Exception {
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


	/**
	 * @return the pairSimilarityMap
	 */
	public Map<SVMPoint, Double> getPairSimilarityMap() {
		return pairSimilarityMap;
	}

	/**
	 * @param pairSimilarityMap the pairSimilarityMap to set
	 */
	public void setPairSimilarityMap(Map<SVMPoint, Double> pairSimilarityMap) {
		this.pairSimilarityMap = pairSimilarityMap;
	}
}