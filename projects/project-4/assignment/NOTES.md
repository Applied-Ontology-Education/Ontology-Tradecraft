Project 4

I broke the work up into chunks:
 1.   take time to understnad the ask and review the various files and plan the work (what the various requirements state- design pattern, IRIs,; )
   
2.    understand what the data is in the files.  Where all of the files exist and where new files need to go.  backup plans, inconsistency in the files, data value issues, null values, naming issues, etc
3.  load the data and process into a common model (canonical)
    determine how to handle the time zone data (the data appeared the same but different -- one file showed local time while the other UTC.  WHat local means in unknown, with 4 hour difference went with et.    
4. Normalize the data (Chiller-3 versus chiller etc)
    
5. Ontology    
    create my ontology file (added the buffalo center location and the stream id--
    
6. RDFLIB file creation - leverage content from the readme file; kept close to the langauge from the design patterns; reference new ontology and BFO and CCO)
        ran test and got back 11 rows with 111 triples.
        received an error on a union message and referenced input from Claire about brackets.-- asking about updating the project4-test.yml file to add the brackets.

Made update to the project4test yaml file to add the brackets and will run a test prior to the next few steps        

SPARQL Work
no duplicate labels has a script
min 1 title - for the ontology file test that there is a title and answer set is filter where not exists -- ran a test to validate
min 1 def with en - add skos; copy script from prior and update.  this will need to be at an element level for classes and obj prop
    -- great when vs code prompts you for completion but not great when what it prompts is not right. 
no duplicate definition - copied no duplicate labels and updated
no whitespace in names - added how to test for whitespaces
all classes have a BFO parent -- more research online to develop this query on the parents
subclass circularity --> no class can be its own ancestor and a + after subclass of will cycle through
no self referential --> a label should not be used in the definition


SHACL Work
do not like - notes are in the file.  


------------

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



TTL Files:
Used Boiler as a starter 
identified the IRIs of the items needed for the design pattern
Went into Protege to create the ontology.
    Device subclasses
    For the boiler device type - had 4 measurement units to support the 4 readings

For step 6