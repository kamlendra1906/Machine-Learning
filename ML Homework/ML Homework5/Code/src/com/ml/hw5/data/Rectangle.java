package com.ml.hw5.data;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.ml.hw5.stats.ImageRectangleFeatures;

/**
 * 
 * @author kkumar
 *
 */
public class Rectangle {

	private Point topLeft;
	private Point bottomRight;
	private Point topRight;
	private Point bottomLeft;
	private Point halfLeft;
	private Point halfRight;
	private Point halfTop;
	private Point halfBottom;
	private Map<Image, ImageRectangleFeatures> imageBlackCountMap;

	public Rectangle(Point p1, Point p2) {
		this.topLeft = p1;
		this.bottomRight = p2;
		
		this.topRight = new Point(bottomRight.getX() - this.getHeight(), bottomRight.getY());
		this.bottomLeft =  new Point(topLeft.getX() + this.getHeight(), topLeft.getY());
		
		this.halfLeft =  new Point(topLeft.getX() + this.getHeight()/2 , topLeft.getY());
		this.halfRight = new Point((topRight.getX() + bottomRight.getX())/2, topRight.getY()); 
		
		this.halfTop = new Point(topLeft.getX(), topLeft.getY() + this.getWidth()/2);
		this.halfBottom = new Point(bottomLeft.getX() , bottomLeft.getY() + this.getWidth()/2);
		
		this.imageBlackCountMap = new HashMap<Image, ImageRectangleFeatures>();
	}
	
	public void calculateBlackCountForAllImages(Map<Image, Map<Point, Integer>> imagePointsBlackCountMap){
		for(Entry<Image, Map<Point, Integer>> e : imagePointsBlackCountMap.entrySet()){
			
			Image img = (Image) e.getKey();
			Map<Point, Integer> pointBlackCountMap = (Map<Point, Integer>) e.getValue();
			ImageRectangleFeatures imageBlackCount = new ImageRectangleFeatures();
			
			int totalBlackCount =  countInsidePoints(topLeft, topRight, bottomLeft, bottomRight, pointBlackCountMap);
			
			int verticalTopCount =  countInsidePoints(topLeft, topRight, halfLeft, halfRight, pointBlackCountMap);
			int verticalBottomCount =  countInsidePoints(halfLeft, halfRight, bottomLeft, bottomRight, pointBlackCountMap);
			
			int horizontalLeftCount =  countInsidePoints(topLeft, halfTop, bottomLeft, halfBottom, pointBlackCountMap);
			int horizontalRightCount =  countInsidePoints(halfTop, topRight, halfBottom, bottomRight, pointBlackCountMap);
			
			imageBlackCount.setTotalBlackCount(totalBlackCount);
			imageBlackCount.setVecticalCount(verticalTopCount - verticalBottomCount);
			imageBlackCount.setHorizontalCount(horizontalLeftCount - horizontalRightCount);
			imageBlackCountMap.put(img, imageBlackCount);
		}
	}
	
	private int countInsidePoints(Point topLeft, Point topRight, Point bottomLeft, Point bottomRight, Map<Point, Integer> pointBlackCountMap) {
		int blackTopLeft = pointBlackCountMap.get(topLeft);
		int blackTopRight = pointBlackCountMap.get(topRight);
		
		int blackBottomLeft = pointBlackCountMap.get(bottomLeft);
		int blackBottomRight = pointBlackCountMap.get(bottomRight);
		
		int blackCount = blackBottomRight - blackTopRight - blackBottomLeft + blackTopLeft;
		return blackCount;
	}

	public int getHeight(){
		return - topLeft.getX() + bottomRight.getX();
	}
	
	public int getWidth(){
		return bottomRight.getY() - topLeft.getY();
	}

	@Override
	public String toString() {
		return "[topLeft=" + topLeft + ", bottomRight=" + bottomRight
				+ "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((bottomRight == null) ? 0 : bottomRight.hashCode());
		result = prime * result + ((topLeft == null) ? 0 : topLeft.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Rectangle other = (Rectangle) obj;
		if (bottomRight == null) {
			if (other.bottomRight != null)
				return false;
		} else if (!bottomRight.equals(other.bottomRight))
			return false;
		if (topLeft == null) {
			if (other.topLeft != null)
				return false;
		} else if (!topLeft.equals(other.topLeft))
			return false;
		return true;
	}

	public ImageRectangleFeatures getBlackCountForImg(Image img) {
		return imageBlackCountMap.get(img);
	}	
}