CREATE TABLE Covid19TrackingFactTable(
	Onset_date_key INTEGER,			--PK/FK
	Reported_date_key INTEGER,		--PK/FK
	Test_date_key INTEGER,			--PK/FK
	Specimen_date_key INTEGER,		--PK/FK
	Patient_key INTEGER,			--PK/FK
	PHU_Location_key INTEGER,		--PK/FK
	Mobility_key INTEGER,			--PK/FK
	Special_Measures_key INTEGER,	--PK/FK
	Wealth_key INTEGER,				--PK/FK
	Number_Resolved INTEGER,
	Number_Unresolved INTEGER,
	Number_Fatal INTEGER,
	PRIMARY KEY (Onset_date_key,
				 Reported_date_key,
				 Test_date_key,
				 Specimen_date_key,
				 Patient_key,
				 PHU_Location_key,
				 Mobility_key,
				 Special_Measures_key,
				 Wealth_key),
	FOREIGN KEY(Onset_date_key)
	REFERENCES Onset_Date_dimension (Onset_date_key),
	FOREIGN KEY(Reported_date_key)
	REFERENCES Reported_Date_dimension (Reported_date_key),
	FOREIGN KEY(Test_date_key)
	REFERENCES Test_Date_dimension (Test_date_key),
	FOREIGN KEY(Specimen_date_key)
	REFERENCES Specimen_Date_dimension (Specimen_date_key),
	FOREIGN KEY(Patient_key)
	REFERENCES Patient_dimension (Patient_key),
	FOREIGN KEY(PHU_Location_key)
	REFERENCES PHU_Location_dimension (PHU_Location_key),
	FOREIGN KEY(Mobility_key)
	REFERENCES Mobility_dimension (Mobility_key),
	FOREIGN KEY(Special_Measures_key)
	REFERENCES Special_Measures_dimension (Special_Measures_key),
	FOREIGN KEY(Wealth_key)
	REFERENCES Wealth_dimension (Wealth_key)
);