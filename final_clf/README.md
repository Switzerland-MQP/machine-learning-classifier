# Final Pipeline Instructions
## Setup
In a command line:
1. Make sure Python 3.x is installed
2. `pip install pipeline_module_name`

## Pipeline Overview
This pipeline works in two main steps. The first step is the optical character recognition, which takes an input direct within the command line. The input directory is the directory that contains the files you would like to be classified within our system. These files can consist of many different types. After the OCR is ran, a hidden output directory will hold a copy of the original files that have been converted into text files. This is done because our classification system can only handle text documents. 

The second main step is the classification of the different files based on the personally identifiable information within them. Our system will take in the hidden directory that contains the text files, and predict the types of personally identifiable information within them. From here, the original files will be placed within directories that relate to these categories: Non Personal, Personal, and Sensitive Personal. Within each subdirectory, there will be an overview document that explains which types of PII each document contains. 

## How to Run
### OCR
In command line run:
- `python nameofOcr.py inputDirectory` 

### Classification
In command line run:
- `python nameOfClass.py outputDirectory`

# Data Labeling Instructions
If you would like to train a model further train your own model, then more data labeling and model training has to occur. This can be done a few ways when used within our system, and has to be done on the text versions of the documents. If you would like to find these text versions, after running the OCR on new training data find XXXXX within your file system. 

## Document Level Labeling
Document level labeling is used to train the models that classify documents into the  Non-Personal, Personal and Sensitive Personal Data categories. To label more training data:
1. Create three directories, one for each: Non-Personal, Personal, Sensitive Personal data
2. Identify category of PII and copy new training data into the corresponding directory


## Line by Line Level Labeling
Line by line level labeling is done to help predict which PII category types are within a certain document, and is also used to help determine where this information in the document lies. This labeling is done on using custom HTML tags that we have created:
1. Personal
   - \<name\>
   - \<id-number\>
     - Health care id numbers, Social Security Numbers, bank account numbers
   - \<location\>
     - current location, personal address, work address if it can be related back to a certain individual clearly
   - \<online-id\>
     - Email, Fax
   - \<dob\>
     - Date of Birth
   - \<phone\>
     - Personal phone number or someone’s personal work number
   - \<professional\>
     - Profession, Workplace, Education, what someone did at a certain job
2. Sensitive Personal
   - \<criminal\>	
     - Information relating back to someone’s criminal history or showing that a person has no criminal history
   - \<origin\>
     - Birthplace, ethnicity, nationality
   - \<health\>
     - Disabilities, hospital visits, health insurance claims
### Tips and Tricks
- A line is not <professional> if it contains the skills of a person. For example, if they have a section in a CV that only lists out skills, this does not count and will throw off the model. However, if they list a job they had and what they did at that job, then this whole area can be considered <professional>
- When using multiple labels in multiple lines (i.e. \<name_phone\>) make sure that each line contained within the HTML tag contains each type of PII that is within the tag
- At times the optimal character recognition will make a copied text file of a document look obscure. If the information is still readable, then tag it as such. However, if it is very obscure, then do not tag it. For example: Name: *Herrrn Smith* is okay to label. Name: *Haaahhrn Sm0th* is not okay to label
- Run `cat.sh` after you finish labeling to check that there is no mistakes, this will generate a file parse-errors.txt which will say which documents have errors and why (it will also print this out to the terminal)

### Examples
**Base Case**
\<name\>\<data\>My name is John Smith\</data\>\</name\>

**Multi-line Case**
\<location\>\<data\>StreetCity,
 State Zip Code\</data\>\</location\>

**Multiple Data Tags Related in a Line**
\<name_phone\>\<data\>Name: John Smith | Phone Number: +1(111)1111\</data\>\</name_phone\>

**Multiple Data Tags Related in Multiple Lines**
\<name_phone\>\<data\>Name: John Smith | Phone Number: +1(111)1111
Name: Jane Doe | Phone Number: +1(222)2222\</data\>\</name_phone\>




