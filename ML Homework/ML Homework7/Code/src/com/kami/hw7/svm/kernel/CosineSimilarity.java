/**
 * 
 */
package com.kami.hw7.svm.kernel;

import java.util.Map;

import com.ml.hw7.data.Data;
import com.ml.hw7.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class CosineSimilarity implements SimilarityStrategy {

	/* (non-Javadoc)
	 * @see com.kami.hw7.svm.kernel.SimilarityStrategy#evaluateSimilarity(com.ml.hw7.data.Data, com.ml.hw7.data.Data, java.util.Map)
	 */
	@Override
	public double evaluateSimilarity(Data x1, Data x2, Map<String, Object> additionalData) throws Exception {
		double dotProduct = SVMUtil.calculateDotProduct(x1, x2);
		double normX1 = SVMUtil.calculateNormOfData(x1);
		double normX2 = SVMUtil.calculateNormOfData(x2);
		return dotProduct/(normX1 * normX2);
	}
}