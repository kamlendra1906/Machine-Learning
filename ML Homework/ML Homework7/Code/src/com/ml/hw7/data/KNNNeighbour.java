/**
 * 
 */
package com.ml.hw7.data;

/**
 * @author kkumar
 *
 */
public class KNNNeighbour implements Comparable<KNNNeighbour> {
	
	private double similarity;
	private Data point;
	
	public KNNNeighbour() {
	}
	
	public KNNNeighbour(double similarity, Data point) {
		this.similarity = similarity;
		this.point = point;
	}

	@Override
	public int compareTo(KNNNeighbour o) {
		if(this.similarity == o.getSimilarity()) {
			return 0;
		}
		if(this.similarity > o.getSimilarity()) {
			return -1;
		}
		return 1;
	}

	/**
	 * @return the similarity
	 */
	public double getSimilarity() {
		return similarity;
	}

	/**
	 * @param similarity the similarity to set
	 */
	public void setSimilarity(double similarity) {
		this.similarity = similarity;
	}

	/**
	 * @return the point
	 */
	public Data getPoint() {
		return point;
	}

	/**
	 * @param point the point to set
	 */
	public void setPoint(Data point) {
		this.point = point;
	}

}
