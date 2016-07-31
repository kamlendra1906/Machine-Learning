/**
 * 
 */
package com.ml.hw6.feature;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Kamlendra Kumar
 *
 */
public class FeatureReader {
	
	public static List<Feature> getFeaturesList(String fileName) throws Exception {
		
		BufferedReader fReader = null;
		List<Feature> features = null;
		try {
			fReader = new BufferedReader(new FileReader(fileName));
			
			features = new ArrayList<Feature>();
			
			while(true){
				String line = fReader.readLine();
				Feature feature = null;
				
				if(line == null){
					break;		
				}
				
				if(line.trim() == ""){
					break;
				}
				
				String[] values = line.trim().split(":");
				if(values[1].contains("continuous")) {
					feature = new Feature(values[0].trim(), Feature.NUMERICAL);
				} else {
					feature = new Feature(values[0].trim(), Feature.LABEL);
				}
				features.add(feature);
			}
		} finally {
			fReader.close();
		}
		return features;
	}
	
	public static List<Feature> getFeatureForHW5(String fileName) throws Exception {
		BufferedReader reader = null;
		List<Feature> features = new ArrayList<Feature>();
		
		try {
			reader = new BufferedReader(new FileReader(fileName));
			String singleLine = reader.readLine();
			
			String delims = "\\s+|,";
			String[] values = singleLine.trim().split(delims);
			
			int featureCount = values.length;

			for(int i=0; i < featureCount; i++) {
				features.add(new Feature("feature"+i, Feature.NUMERICAL));
			}
			features.add(new Feature("label", Feature.LABEL));
		} catch(Exception e) {
			throw e;
		} finally {
			reader.close();
		}
		return features;
	}
}