package com.covid19.datastaging;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.opencsv.CSVWriter;


public class MissingValueHandler {

	public static String[][] csvFileReader(String filePath) throws Exception{
		Scanner sc = new Scanner(new File(filePath));
		
		List<String[]> lines = new ArrayList<String[]>();
		int length = 0;
		if (sc.hasNextLine()) {
			String[] title = sc.nextLine().split(",");
			length = title.length;
			lines.add(title);
		}
		String[] arr,nl;
		while (sc.hasNextLine()) {
			arr = new String[length];
			nl = sc.nextLine().split(",");
			for (int i = 0; i<length;i++) {
				if(i<nl.length && !nl[i].isEmpty()) {
					arr[i]=nl[i];
				} else {
					arr[i] = "N/A";
				}
			}
		    lines.add(arr);
		}

		//System.out.println(length);
		String[][] array = new String[lines.size()][length];
		lines.toArray(array);
		
		sc.close();
		return array;
	}
	
	public static boolean isNumeric(String str) {
		return str.matches("-?\\d+(\\.\\d+)?");  //match a number with optional '-' and decimal.
	}
	
	public static String[][] fillNum(String[][] table) {
		boolean[] isInt = new boolean[table[0].length]; 
		isInt[0] = false;
		for (int i = 1; i<table[0].length; i++) {
			for (int j = 1; j<table.length; j++) {
				isInt[i] = false;
				if (table[j][i]!="N/A") {
					if (isNumeric(table[j][i])) {
						isInt[i] = true;
						//System.out.println("2" + isInt[i]);
					} else {
						//System.out.println("3" + isInt[i]);
					}
					break;
				}
			}
			//System.out.println("1");
		}
		
		//for (int i = 0; i<table[0].length; i++) System.out.println(isInt[i]);
		int average, counter;
		for (int i = 1; i<table[0].length; i++) {
			if (isInt[i]) {
				average = 0; counter = 0;
				for (int j = 1; j<table.length; j++) {
					if (table[j][i] != "N/A") {
						average += Integer.parseInt(table[j][i]);
						counter++;
					}
				}
				for (int j = 1; j<table.length; j++) {
					if (table[j][i] == "N/A") {
						table[j][i] = "" + (average / counter);
					}
				}
			}
		}
		return table;
	}
	
	public static void csvOutputter(String[][] table) {
		
		File file = new File("./output/result.csv"); 
	    try { 
	        // create FileWriter object with file as parameter 
	        FileWriter outputfile = new FileWriter(file); 
	  
	        // create CSVWriter object filewriter object as parameter 
	        CSVWriter writer = new CSVWriter(outputfile); 
	  
	        for (int i = 0; i < table.length; i++) {
	        	writer.writeNext(table[i]);
	        	for (int j = 0; j < table[i].length; j++) {
	        		if(table[i][j].isEmpty()) {
	        			System.out.print("N/A ");
	        		}
	        		else System.out.print(table[i][j]+" ");
	        	}
	        	System.out.println();
	        }
	        
			/*
			 * // adding header to csv String[] header = { "Name", "Class", "Marks" };
			 * writer.writeNext(header);
			 * 
			 * // add data to csv String[] data1 = { "Aman", "10", "620" };
			 * writer.writeNext(data1); String[] data2 = { "Suraj", "10", "630" };
			 * writer.writeNext(data2);
			 */
	  
	        // closing writer connection 
	        writer.close(); 
	    } 
	    catch (IOException e) { 
	        // TODO Auto-generated catch block 
	        e.printStackTrace(); 
	    } 
		
	}
	
	public static void main(String[] args) {
		String[][] csvData = null;
		String filepath = "";
		if (args.length == 0) {
			filepath = "./resources/Mobility_dimension_CSV.csv";
		}
		else if (args.length >= 1) {
			filepath = args[0];
		}
		try {
			csvData = csvFileReader (filepath);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//csvOutputter(csvData);
		
		String[][] filledCSV = fillNum(csvData);
		
		csvOutputter(filledCSV);
		
		/*
		 * boolean a = isNumeric("1"); boolean b = isNumeric("b"); boolean c =
		 * isNumeric("-1"); System.out.println(a); System.out.println(b);
		 * System.out.println(c);
		 */
		 
	}

}
