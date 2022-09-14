# Assignment 1 &middot;

`first` assignment of Machine Learning course (CS60050) offered in Autumn semester 2022, Department of CSE, IIT Kharagpur.

## Getting started

Read the assignment problem statement from [Assignment_1.pdf](/Asgn1/Assignment%201_A.pdf)

Dataset is provided in the file [Dataset_A.csv](/Asgn1/Dataset_A.csv) and its description in [Dataset_A_Description.pdf](/Asgn1/Dataset_A-Description.pdf)

Python version information-  

```shell
Python 3.10.5
```

- Install required python packages-

```shell
pip install -r requirements.txt
```

(In case of an error, notice the required packages for running the files are- `numpy`, `matplotlib`, `pandas`). Install them individually if the above command fails (version conflict)

- Run [DataCleaning.py](/Asgn1/DataCleaning.py) to clean and categorically encode the dataset and produce [cleanedData.csv](/Asgn1/cleanedData.csv)

```shell
python3 DataCleaning.py
```

## Solution

- [Q1.py](/Asgn1/Q1.py) is the submission file for the first question
- [Q2.py](/Asgn1/Q2.py) is the submission file for the second question

- Run these files individually to get the output on terminal and output files.

```shell
python3 Q1.py
python3 Q2.py
```

- [DecisionTree.txt](/Asgn1/DecisionTree.txt) contains the final decision tree.
- [accuracy_VS_depth.png](/Asgn1/accuracy_VS_depth.png) conatins the plot of accuracy vs depth for the decision tree.
- [NaiveBayes.txt](/Asgn1/NaiveBayes.txt) contains the results for Naive Bayes classifier.
