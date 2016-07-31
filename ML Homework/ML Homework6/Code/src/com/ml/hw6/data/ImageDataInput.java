/**
 * 
 */
package com.ml.hw6.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.ml.hw6.feature.Feature;
import com.ml.hw6.stats.ImageRectangleFeatures;

/**
 * @author kkumar
 *
 */
public class ImageDataInput {

	private List<Rectangle> rectangles;
	
	public ImageDataInput(int numRectangles) {
		this.rectangles = generateRandomRectangles(numRectangles);
	}
	
	public DataSet getImageDataSet(List<Image> images) throws Exception {
		Map<Image, Map<Point, Integer>> imagePointsBlackCountMap = getImagePointsBlackCount(images);
		populateRectangleFeatures(imagePointsBlackCountMap);
		return getDataSet(images);
	}

	private DataSet getDataSet(List<Image> images) throws Exception {
		List<Feature> features = generateFeatures(rectangles.size()*2);
		
		int labelIndex = features.size() -1;
		int dataValueArraySize = features.size();
		
		DataSet dataset = new DataSet(labelIndex, features);
		
		for(Image image : images) {
			Data data = new Data(dataValueArraySize);
			int feature=0;
			
			for(Rectangle rectangle : rectangles) {
				ImageRectangleFeatures imageRectangleFeatures = rectangle.getBlackCountForImg(image);
				if(imageRectangleFeatures != null){
					data.setFeatureValue(feature++, imageRectangleFeatures.getVecticalCount());
					data.setFeatureValue(feature++, imageRectangleFeatures.getHorizontalCount());
				}
			}
			
			data.setDataSet(dataset);
			data.setLabelValue(image.getLabel());
			dataset.addData(data);
		}
		return dataset;
	}

	private List<Feature> generateFeatures(int featureSize) {
		List<Feature> features = new ArrayList<Feature>();
		for(int i=0; i < featureSize; i++) {
			features.add(new Feature("feature"+i, Feature.NUMERICAL));
		}
		features.add(new Feature("Class", Feature.LABEL));
		return features;
	}

	private void populateRectangleFeatures(Map<Image, Map<Point, Integer>> imagePointsBlackCountMap) {
		for(Rectangle rectangle : this.rectangles) {
			rectangle.calculateBlackCountForAllImages(imagePointsBlackCountMap);
		}
	}

	private Map<Image, Map<Point, Integer>> getImagePointsBlackCount(List<Image> images) {
		Map<Image, Map<Point, Integer>> imagePointsBlackCountMap = new HashMap<Image, Map<Point,Integer>>();
		
		for(Image image : images) {
			Map<Point, Integer> pointsBlackCountMap = getPointsBlackCount(image);
			imagePointsBlackCountMap.put(image, pointsBlackCountMap);
		}
		return imagePointsBlackCountMap;
	}

	private Map<Point, Integer> getPointsBlackCount(Image image) {
		Map<Point, Integer> map = new HashMap<Point, Integer>();
		
		for (int col = 0; col < image.getNumCols(); col++) {
			for (int row = 0; row < image.getNumRows(); row++) {
				
				Point pij = new Point(row, col);
				Point pi_1j = new Point(row - 1, col);
				Point pi_1j_1 = new Point(row - 1, col - 1);
				Point pij_1 = new Point(row, col - 1);
				
				int blacki_1j = map.get(pi_1j) == null ? 0 : map.get(pi_1j);
				int blacki_1j_1 = map.get(pi_1j_1) == null ? 0 : map.get(pi_1j_1);
				int blackij_1 = map.get(pij_1) == null ? 0 : map.get(pij_1);
				int blackij = blackij_1 + blacki_1j - blacki_1j_1 + image.getImagePixel(row, col);				

				map.put(pij, blackij);
			}
		}
		return map;
	}
	
	public List<Rectangle> generateRandomRectangles(int numRectangles){
		List<Rectangle> rectangles = new ArrayList<Rectangle>();
		while(rectangles.size() < numRectangles){
			Point p1 = generateRandomPoint();
			Point p2 = generateRandomPoint(p1);
			
			Rectangle rect = new Rectangle(p1, p2);
			rectangles.add(rect);
		}
		return rectangles;
	}

	private Point generateRandomPoint(Point p1) {
		int oX = p1.getX();
		int oY = p1.getY();
		
		int x = (int) getRandomValue(oX + 5, 27);
		int y = (int) getRandomValue(oY + 5, 27);
		
		return (new Point(x, y));
	}

	private Point generateRandomPoint() {
		int x = (int) getRandomValue(0, 27 - 5);
		int y = (int) getRandomValue(0, 27 - 5);
		
		return (new Point(x,y));
	}
	
	private double getRandomValue(double start, double end) {
		Random random = new Random();
		double range = end - start;
		double fraction = range * random.nextDouble();
		return fraction + start;
	}
}