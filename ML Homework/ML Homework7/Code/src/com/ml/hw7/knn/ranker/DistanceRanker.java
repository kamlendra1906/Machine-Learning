/**
 * 
 */
package com.ml.hw7.knn.ranker;

import java.util.Comparator;

import com.ml.hw7.data.KNNNeighbour;

/**
 * @author kkumar
 *
 */
public class DistanceRanker implements Comparator<KNNNeighbour> {

	@Override
	public int compare(KNNNeighbour o1, KNNNeighbour o2) {
		if(o1.getSimilarity() == o2.getSimilarity()) {
			return 0;
		}
		if(o1.getSimilarity() < o2.getSimilarity()) {
			return -1;
		}
		return 1;
	}
}
