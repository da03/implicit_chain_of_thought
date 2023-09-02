import sys
import os

import json
import re

# Open the input file for reading and output file for writing
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    # Read each line from the input file
    for line in infile:
        # Parse the JSON line into a Python dictionary
        data = json.loads(line.strip())
        
        # Get the 'question' field from the dictionary
        question = data.get("question", "")
        
        # Get the 'answer' field from the dictionary and find all formulas within <<>>
        answer = data.get("answer", "")
        formulas = re.findall(r'<<([^>>]+)>>', answer)
        
        # Join the formulas with spaces
        formulas_str = ' '.join(formulas)
        
        # Construct the desired line format
        formatted_line = f"{question}||{formulas_str}"
        
        # Write the formatted line to the output file
        outfile.write(formatted_line + "\n")



{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?", "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"}



3 5 4 1 1 * 2 1 4 6 5||6 0 9 2 2 0 + 0 3 5 4 1 1 0 ( 6 3 4 7 3 1 0 ) + 0 0 2 1 8 5 4 0 ( 6 3 6 8 1 7 4 0 ) + 0 0 0 8 1 7 8 6 0 ( 6 3 6 6 3 4 3 7 0 ) + 0 0 0 0 5 6 2 7 5 0 #### 6 3 6 6 8 0 6 4 6 0
