package com.ml.hw7.data;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This class implements a reader for the MNIST dataset of handwritten digits.
 * The dataset is found at http://yann.lecun.com/exdb/mnist/.
 * 
 * @author Gabe Johnson <johnsogg@cmu.edu>
 */

public class MNISTReader {
	
	public static List<Image> getImages(String labelFilename, String imageFileName) throws IOException {
		List<Image> images = new ArrayList<Image>();
		DataInputStream imageStream = null;
		DataInputStream labelStream = null;
		try {
			labelStream = new DataInputStream(new FileInputStream(labelFilename));
			imageStream = new DataInputStream(new FileInputStream(imageFileName));
			
			int magicNumber = labelStream.readInt();
			if (magicNumber != 2049) {
				System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
				System.exit(0);
			}
			magicNumber = imageStream.readInt();
			if (magicNumber != 2051) {
				System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
				System.exit(0);
			}
			
			int numLabels = labelStream.readInt();
			int numImages = imageStream.readInt();
			int numRows = imageStream.readInt();
			int numCols = imageStream.readInt();
			
			if (numLabels != numImages) {
				System.err.println("Image file and label file do not contain the same number of entries.");
				System.err.println("  Label file contains: " + numLabels);
				System.err.println("  Image file contains: " + numImages);
				System.exit(0);
			}
	
			long start = System.currentTimeMillis();
			int numLabelsRead = 0;
			int numImagesRead = 0;
			while (labelStream.available() > 0 && numLabelsRead < numLabels) {
				int label = labelStream.readByte();
				numLabelsRead++;
				int[][] image = new int[numCols][numRows];
				for (int colIdx = 0; colIdx < numCols; colIdx++) {
					for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
						image[colIdx][rowIdx] = imageStream.readUnsignedByte() == 0 ? 0 : 1;
//						System.out.print(image[colIdx][rowIdx]);
					}
//					System.out.println();
				}
//				System.out.println();
//				System.out.println(label);
//				System.out.println();
				numImagesRead++;
				
				Image img = new Image(numRows, numCols, numImagesRead);
				img.setImage(image);
				img.setLabel(label);
				images.add(img);
				
				/*if(numImagesRead == 10000){
					break;
				}*/
			}
//			System.out.println();
			long end = System.currentTimeMillis();
			long elapsed = end - start;
			long minutes = elapsed / (1000 * 60);
			long seconds = (elapsed / 1000) - (minutes * 60);
			System.out.println("Read " + numLabelsRead + " samples in " + minutes + " m " + seconds + " s ");
		} finally {
			labelStream.close();
			imageStream.close();
		}
		return images;
	}

}