package com.ml.hw5.data;

import java.util.Arrays;

/**
 * @author Kamlendra Kumar
 */
public class Image {

	private int[][] image;
	private int label;
	private int numRows;
	private int numCols;
	private int id;
	
	public Image(int r, int c, int id) {
		this.image = new int[r][c];
		this.numRows = r;
		this.numCols = c;
		this.setId(id);
	}
	
	public void setImage(int[][] m){
		this.image = m;
	}
	
	public void setLabel(int l){
		this.label = l;
	}
	
	public int getImagePixel(int i, int j){
		return image[i][j];
	}

	@Override
	public String toString() {
		for (int col = 0; col < numCols; col++) {
			for (int row = 0; row < numRows; row++) {
				System.out.print(image[col][row]);
			}
			System.out.println();
		}
		return "Image";
	}

	@Override
	public int hashCode() {
		final int prime = 131;
		int result = 1;
		result = prime * result + Arrays.hashCode(image);
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
		Image other = (Image) obj;
		if (!Arrays.deepEquals(image, other.image))
			return false;
		return true;
	}

	public double getLabel() {
		return (double) label;
	}

	/**
	 * @return the numRows
	 */
	public int getNumRows() {
		return numRows;
	}

	/**
	 * @param numRows the numRows to set
	 */
	public void setNumRows(int numRows) {
		this.numRows = numRows;
	}

	/**
	 * @return the numCols
	 */
	public int getNumCols() {
		return numCols;
	}

	/**
	 * @param numCols the numCols to set
	 */
	public void setNumCols(int numCols) {
		this.numCols = numCols;
	}

	/**
	 * @return the image
	 */
	public int[][] getImage() {
		return image;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}	
}