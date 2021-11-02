# ISWC2021_SMART
Link of Official Website:
https://smart-task.github.io/2021/

Brief introdution of this challenge:
Answer type classification plays a key role in question answering, it aims to do, because identify the  expected answer type so that filter irrelevant information, which notably increases the performance of QA systems. A granular answer type classification is possible with popular Semantic Web ontologies such as DBepdia (~760 classes) and Wikidata (~50K classes). In this challenge, given a question in natural language, the task is to predict type of the answer using a set of candidates from a target ontology.

We present our appoarch to predict types from an ontology for questions. The workflow of our solution is below:

![Fig3](https://user-images.githubusercontent.com/72255811/139656510-3377afaa-ab09-41d8-82ff-151a36b050cf.jpg)

How to use our code? 

data_analysis.ipynb: Simple data analysis. 

Step1: run BERT_classifier_5classes.ipynb to get category classification model.

Step2: run bert_resources_top.py to get general answer type rediction model for resource category. 

Step3: run bert_resources_bottom.py to get specific answer type rediction model for resource category. 

Step4: run Combined_prediction.ipynb to combine all the models to predicae the final results. 

 
