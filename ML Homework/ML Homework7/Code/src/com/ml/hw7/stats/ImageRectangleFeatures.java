package com.ml.hw7.stats;

public class ImageRectangleFeatures {
	
	private int totalBlackCount;
	private int vecticalCount;
	private int horizontalCount;
	
	public ImageRectangleFeatures() {
		this.totalBlackCount = 0;
		this.vecticalCount = 0;
		this.horizontalCount = 0;;
	}
	
	public int getTotalBlackCount() {
		return totalBlackCount;
	}

	public void setTotalBlackCount(int totalBlackCount) {
		this.totalBlackCount = totalBlackCount;
	}

	public int getVecticalCount() {
		return vecticalCount;
	}

	public void setVecticalCount(int vecticalCount) {
		this.vecticalCount = vecticalCount;
	}

	public int getHorizontalCount() {
		return horizontalCount;
	}

	public void setHorizontalCount(int horizontalCount) {
		this.horizontalCount = horizontalCount;
	}
}