/**
 * 
 */
package com.ml.hw6.data;

/**
 * @author kkumar
 *
 */
public class SVMPoint {

	private int x1;
	private int x2;
	
	public SVMPoint(int x1, int x2) {
		this.x1 = x1;
		this.x2 = x2;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(this == obj) {
			return true;
		}
		if(obj == null) {
			return false;
		}
		if(!(obj instanceof SVMPoint)) {
			return false;
		}
		SVMPoint other = (SVMPoint) obj;
		if((x1 == other.x1 && x2 == other.x2) ||
				x1 == other.x2 && x2 == other.x1) {
			return true;
		}
		return false;
	}
	
	@Override
	public int hashCode() {
		int prime = 131;
		int result = Math.abs(x1-x2);
		result+= prime * result;
		return result;
	}
}
