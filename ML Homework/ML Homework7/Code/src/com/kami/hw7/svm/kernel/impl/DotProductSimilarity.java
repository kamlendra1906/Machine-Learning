/**
 * 
 */
package com.kami.hw7.svm.kernel.impl;

import java.util.Map;

import com.kami.hw7.svm.kernel.SimilarityStrategy;
import com.ml.hw7.data.Data;
import com.ml.hw7.util.SVMUtil;

/**
 * @author kkumar
 *
 */
public class DotProductSimilarity implements SimilarityStrategy {

	/* (non-Javadoc)
	 * @see com.kami.hw7.svm.kernel.SimilarityStrategy#evaluateSimilarity(com.ml.hw7.data.Data, com.ml.hw7.data.Data, java.util.Map)
	 */
	@Override
	public double evaluateSimilarity(Data x1, Data x2, Map<String, Object> additionalData) throws Exception {
		return SVMUtil.calculateDotProduct(x1, x2);
	}
}
