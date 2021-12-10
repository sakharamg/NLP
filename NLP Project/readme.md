# Project : Tweet Classification for Crisis Response  

## How to run code

### Using Google Colab  

We recommend you to run the .ipynb files on Google Colab(and use GPU for BERT)  

1) BERT_with_ZeroShot_demonstration.ipynb: SVM with BERT as embeddings. The file also shows zero-shot learning for SVM with BERT and also has a test function for passing a custom sentence  
Pre-requisites: The code should automatically install the pre-req on Google Colab, may require runtime restart  
Colab Link(IITB Login Required): https://colab.research.google.com/drive/1T85sKyEwcX724oiloqlgaWuHDWt3nD0W?usp=sharing       

2) SVM_TFIDF_FeatEngg.ipynb: SVM with TF_IDF and Feature Engineering. The file uses combined data for training  
Pre-requisites: The code should automatically install the pre-req on Google Colab   
Colab Link: https://colab.research.google.com/drive/1wHw05XT9NgRb92bzdriJLzoB0USEW5MN?usp=sharing  

3) Critical_Data_Analysis_Crisis_Data.ipynb: Contains visualizations of critical tweets on Train Data   
Pre-requisites: The code should automatically install the pre-req on Google Colab, may require runtime restart  
Colab Link: https://colab.research.google.com/drive/15JD_FItJ2W9EYmk54VO8UCdld1pqYNxD?usp=sharing        

4) Data_Preprocessing.ipynb: Preprocesses Tweets  
Pre-req: The code should automatically install the pre-req on Google Colab, may require runtime restart   
Data: https://crisisnlp.qcri.org/data/lrec2016/labeled_cf/CrisisNLP_labeled_data_crowdflower_v2.zip

5) Cross_Domain_SVM_TD_DF_Feat_Engg_Analysis.ipynb: Zero Shot Learning Cross Domain Experimentation with SVM TF_IDF  
Pre-req: The code should automatically install the pre-req on Google Colab, may require runtime restart  

For 1,2,3 and 5, data is already uploaded on internet and none of these files are dependent on each other  

## References
  
1) *Sentence Embeddings*:  https://simpletransformers.ai/docs/text-rep-model/  
2) *Tweets Visualization*:  https://towardsdatascience.com/exploratory-data-analysis-for-natural-language-processing-ff0046ab3571  
3) *SVM Sklearn*:  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html  
4) *Display classification report*:  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html  
5) *Crisis NLP Dataset and OOV words*: https://crisisnlp.qcri.org/lrec2016/lrec2016.html  
