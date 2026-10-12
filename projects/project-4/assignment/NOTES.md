Project 4

Extract every "must", file path, and command into the tracker.

Cross-check each one against the grader. Where they disagree (gotchas 1, 8, 11), the grader decides what passes. Log the conflict.

Predict what will fail before you run anything, then run it and compare.



Requirement		File					Pass condition													Status				Evidence
Workflow		workflow.yml			<15 minute run time
Workflow							one newly added file un src/data (one new dataset per push)
Workflow		workflow.yml			python version 3.11 cache pip										should be ok ay using 3.13
Workflow		workflow.yml			pip	 - installed													good				ran script										
Normalized							updated main to support C as optional		(requirement 3 and 4)						initial run 6 rows in A 8 in B and C was skipped
Normalized							6 and 8 records processed to create 11 entries in readings normalized. - (requirement 5)
3 rows were dropped (Requirement 9)- null value of temp for chiller 3 (CSV); null value of psi for pump A1 (CSV) and chiller 3 with no temp value (json)
TS are consistent.  
Chiller 3 looks duplicate and duplicates temperature
Boiler 7 has two temps (one C and one F and has 2 pressure Psi and KPA
Chiller 4 has two readings for temp 

Normalizing the timestamp: - Requirement 7
1. Imported the Zoneinfo  (a built-in Python standard library module that provides a concrete time zone implementation to support the IANA time zone database. It allows you to seamlessly integrate real-world time zones, daylight saving time (DST) shifts, and historical offsets directly into Python's native datetime)
2. Updated the code to set the local time to ET and adjusted for DST.

Normalized the temperature readings Requirement 8
 Valid values are codes like C/F (nothing spelled out)


Trim whitespace - requirement 6 - rows 142 - 145
Compared the script to the requirements.  No change made and confirmed that the Pump A1 has no extra spaces.



ETL Script

Files:  Entity Id = Device Name (difference in name (chiller 3 and chiller-3; (CSV file has the discrepancy)
Kind = Reading Type
Temperature F versus Celcius f
Pressure


Reading type = Temp/Pressure/Voltage/Resistance
Reading value = decimals and nulls
UNit = measurement unit
Time (date time - UTC




Total of 5 devices


Json - Has a site for the reading; has a stream fr the collection/instance of the reading
Time is in local time - not etc (do not know the time zone
Extra entity/device too be read (chiller 4 and circuit 12
Has Celsius and Fahrenheit
Additional reading of voltage
Additional reading of resistance


Censor A Csv
Multiple rows per device (row for temp and for pressure 
Chiller 3 has duplicate temp
Pump A1 has duplicate of pressure

Time is UTC time



Csv

Normalized File


Data Issues

Source		Row			Issues			Decision




Modeling Decisions

Timezone
Unit of Measure
One SDC per artifact or per reading
…


QUestions
