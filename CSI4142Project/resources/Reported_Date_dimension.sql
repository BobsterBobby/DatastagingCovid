CREATE TABLE Reported_Date_dimension(
	Reported_date_key INTEGER,			--PK
	Day INTEGER,		
	Month ENUM('January','February','March','April','May','June','July','August','September','October','November','December'),
    Year INTEGER,
    DayOfTheWeek ENUM('Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
    WeekInYear INTEGER,
    Weekend BOOLEAN,
    Holiday BOOLEAN,
    Season ENUM('Winter','Spring','Summer','Fall'),
    PRIMARY KEY (Reported_date_key)
);