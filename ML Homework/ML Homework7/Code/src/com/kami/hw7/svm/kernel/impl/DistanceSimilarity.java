/**
 * 
 */
package com.kami.hw7.svm.kernel.impl;

import java.util.Map;

import com.kami.hw7.svm.kernel.SimilarityStrategy;
import com.ml.hw7.data.Data;

/**
 * @author kkumar
 *
 */
public class DistanceSimilarity implements SimilarityStrategy {

	/* (non-Javadoc)
	 * @see com.kami.hw7.svm.kernel.SimilarityStrategy#evaluateSimilarity(com.ml.hw7.data.Data, com.ml.hw7.data.Data, java.util.Map)
	 */
	@Override
	public double evaluateSimilarity(Data x1, Data x2, Map<String, Object> additionalData) throws Exception {
		int featureSize = x1.getFeatures().size() -1;
		double vectorDiff = 0;
		for(int i=0; i < featureSize; i++) {
			vectorDiff+= Math.pow((x1.getFeatureValue(i) - x2.getFeatureValue(i)),2);
		}
		return Math.sqrt(vectorDiff);
	}
}