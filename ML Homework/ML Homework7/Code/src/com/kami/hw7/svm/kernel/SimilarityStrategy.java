/**
 * 
 */
package com.kami.hw7.svm.kernel;

import java.util.Map;

import com.ml.hw7.data.Data;

/**
 * @author kkumar
 *
 */
public interface SimilarityStrategy {

	public double evaluateSimilarity(Data x1, Data x2, Map<String, Object> additionalData) throws Exception;
}
