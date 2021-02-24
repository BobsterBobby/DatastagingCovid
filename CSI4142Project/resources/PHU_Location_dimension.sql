CREATE TABLE PHU_Location_dimension(
	PHU_Location_key INTEGER,			--PK
	PHU_Address VARCHAR(40),
    City VARCHAR(20),
    PostalCode VARCHAR(6), --Assuming that the space in the middle of the postal code is taken out
    Province VARCHAR(25),
    PHU_URL URL,
    Latitude FLOAT,
    Longitude FLOAT,
    PRIMARY KEY (PHU_Location_key)
);
