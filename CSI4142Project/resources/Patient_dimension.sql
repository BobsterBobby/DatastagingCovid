CREATE TABLE Patient_dimension(
	Patient_key INTEGER,			--PK
	Sex ENUM('Male','Female','Other'),
    AgeGroup ENUM('0-12','12-17','18-24','25-34','35-44','45-54','55-64','65-74','75-84','85+'),
    AcquisitionGroup VARCHAR(40),
    OutbreakRelated BOOLEAN,
    PRIMARY KEY (Patient_key)
);
