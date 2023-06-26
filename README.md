# Contributing_statement
Identifying contributing statement from document
Input: NLP Trial Data for Training and Published paper for Testing. 
Output: Performance of the NLP/ML model (ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore)

Preprocessing (Code=Preprocess.py) output: Preprocessed_data.csv 
  Step 1: Convert the PDF (published paper) into a text file (1. txt). 
  Step 2: Identify the sentences and tokens of the paper (2. txt). 
  Step 3: Create a .csv file showing the class label for the sentences.
  
Training (Code = Bert.py) 
  Step 4: Apply the Bert classification model: 
    Step 4.1: Initialize the model arguments: 
      Step 4.1.1: Learning rate 
      Step 4.1.2: Train batch size 
      Step 4.1.3: Number of Epochs 
    Step 4.2: Generate a training model: 
      Step 5.1: Apply the model to the .csv file generated in Step 3. 

Testing (Code = Bert.py) output: Contributing_statement.csv
Step 5: For the given pdf (published paper) 
  Step 5.1: Apply Step 1, 2, and 3 to generate .csv preprocessed file. 
  Step 5.2: Apply the trained model of Step 4.2 to the .csv file of Step 5.1. 
  Step 5.3: Store the results F1_score, Confusion metrics (fn, fp, tn, and tp) Step 5.3: Store the contributing statements (class label predicted as 1). 
  
Output: F1_score = 0.44347826086956516, fn = 27, fp = 101, mcc = 0.4124192864140653, tn = 914, tp = 51
