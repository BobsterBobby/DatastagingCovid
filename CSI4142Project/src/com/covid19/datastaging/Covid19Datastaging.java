package com.covid19.datastaging;

import java.io.BufferedReader;
import java.io.FileReader;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class Covid19Datastaging {
	
	public static void main(String[] args) {
		
		String serverURL = args[0];
		String dbUser = args[1];
		String password = args[2];// TODO: add file argument instead of auto chosing 
		
		if (serverURL == null) {
			
		}
		
		System.out.println("Connecting to database...");
				
		Connection c = null;
		Statement stmt = null;
		ResultSet rs = null;
	      try {
	         //Class.forName("org.postgresql.Driver");
	         c = DriverManager
	            .getConnection(serverURL,
	            		dbUser, password);
	         System.out.println("Opened database successfully");
	         
	         stmt = c.createStatement();
	         
	         BufferedReader bufferedReader = new BufferedReader(
                     new FileReader("/resources/FactTableCreate.sql"));
	         
	         StringBuilder sb = new StringBuilder();
	         String line;
	         while ((line = bufferedReader.readLine()) != null)
	         {
	             sb.append(line);
	         }
	         bufferedReader.close();
	         
	         String sql = sb.toString();
	         
	         
	         rs = stmt.executeQuery(sql);
	         
			/*
			 * while ( rs.next() ) {
			 * String employee_name = rs.getString("employee_name");
			 * String manager_name = rs.getString("manager_name"); float salary =
			 * rs.getFloat("employee_salary");
			 * 
			 * System.out.println( "Employee Name = " + employee_name ); System.out.println(
			 * "Manager Name = " + manager_name ); System.out.println( "SALARY = " + salary
			 * ); System.out.println();
			 * }
			 */
	         
	      } catch (Exception e) {
	         e.printStackTrace();
	         System.err.println(e.getClass().getName()+": "+e.getMessage());
	         System.exit(0);
	      } finally {
	    	  try {
	    		  if (rs  != null) {
	    			  rs.close();
	    		  }
	    		  if (stmt  != null) {
	    			  stmt.close();
	    		  }
	    		  if (c  != null) {
	    			  c.close();
	    		  }
	    		  
	    	  } catch ( SQLException e) {
	 	         e.printStackTrace();
	    	  }
	      }
		
		System.out.println("Finished");
		
	}

}
