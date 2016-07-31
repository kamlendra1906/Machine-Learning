/**
 * 
 */
package com.kami.hw7.svm.kernel.impl;

import java.util.HashMap;
import java.util.Map;

import com.kami.hw7.svm.kernel.CosineSimilarity;
import com.kami.hw7.svm.kernel.SVMKernel;
import com.kami.hw7.svm.kernel.SimilarityStrategy;
import com.ml.hw7.data.Data;
import com.ml.hw7.data.SVMPoint;
import com.ml.hw7.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class AbstractKernel implements SVMKernel {
	
	private Map<SVMPoint, Double> pairSimilarityMap;
	private double[][] pairSimilarityArray;
	private SimilarityStrategy similarityStrategy;
	
	public AbstractKernel(int size, int similarityStrategy) {
		pairSimilarityMap = new HashMap<SVMPoint, Double>();
		pairSimilarityArray = new double[size][size];
		for(int i=0;i<size;i++) {
			for(int j=0;j<size;j++) {
				pairSimilarityArray[i][j] = Double.NaN;
			}
		}
		this.similarityStrategy = getSimilarityStrategy(similarityStrategy);
	}

	private SimilarityStrategy getSimilarityStrategy(int similarityStrategy) {
		if(similarityStrategy == SVMUtil.RBF_SIMILARITY) {
			return new RBFSimilarity();
		}
		if(similarityStrategy == SVMUtil.DISTANCE_SIMILARITY) {
			return new DistanceSimilarity();
		}
		if(similarityStrategy == SVMUtil.COSINE_SIMILARITY) {
			return new CosineSimilarity();
		}
		return new DotProductSimilarity();
	}

	@Override
	public double evaluateKernel(Data x1, Data x2, int indexX1, int indexX2, Map<String, Object> additionalData)
			throws Exception {
		double dotProduct = Double.NaN;
		boolean cacheEnabled = (boolean) additionalData.get(SVMUtil.SVM_PARAMETERS_KERNEL_CACHE_ENABLED);
		if(cacheEnabled) {
			dotProduct = pairSimilarityArray[indexX1][indexX2];
			if(Double.isNaN(dotProduct)) {
				dotProduct = similarityStrategy.evaluateSimilarity(x1, x2, additionalData);
				pairSimilarityArray[indexX1][indexX2] = dotProduct;
				pairSimilarityArray[indexX2][indexX1] = dotProduct;
			}
		} else {
			dotProduct = similarityStrategy.evaluateSimilarity(x1, x2, additionalData);
		}
		if(Double.isNaN(dotProduct)) {
			throw new Exception("NaN from Kernal");
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