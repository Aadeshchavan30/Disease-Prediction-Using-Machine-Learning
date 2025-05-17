# Disease-Prediction-Using-Machine-Learning
Disease prediction using machine learning is used in healthcare to provide accurate and early diagnosis based on patient symptoms. We can build predictive models that identify diseases efficiently. In this article, we will explore the end-to-end implementation of such a system.

Step 1: Import Libraries
We will import all the necessary libraries like pandas, Numpy, scipy, matplotlib, seaborn and scikit learn.

Step 2: Reading the dataset
In this step we load the dataset and encode disease labels into numbers and visualize class distribution to check for imbalance. We then use RandomOverSampler to balance the dataset by duplicating minority classes and ensuring all diseases have equal samples for fair and effective model training.
![image](https://github.com/user-attachments/assets/84d18279-a206-4d7f-a319-c557513a5e70)


Step 3: Cross-Validation with Stratified K-Fold
We use Stratified K-Fold Cross-Validation to evaluate three machine learning models. The number of splits is set to 2 to accommodate smaller class sizes
The output shows the evaluation results for three models SVC, Gaussian Naive Bayes and Random Forest using cross-validation. Each model has two accuracy scores: 1.0 and approximately 0.976 indicating consistently high performance across all folds.

Step 4: Training Individual Models and Generating Confusion Matrices
After evaluating the models using cross-validation we train them on the resampled dataset and generate confusion matrix to visualize their performance on the test set.
![image](https://github.com/user-attachments/assets/1b456ab6-031b-49e4-af2d-ba09e146d1a5)
The matrix shows good accuracy with most values along the diagonal meaning the SVM model predicted the correct class most of the time.
![image](https://github.com/user-attachments/assets/b6e77a76-3dc1-448a-b80c-f6db8d00cf6e)
Naive Bayes Accuracy: 37.98%
This matrix shows many off-diagonal values meaning the Naive Bayes model made more errors compared to the SVM. The predictions are less accurate and more spread out across incorrect classes.

Random Forest Classifier
Random Forest Accuracy: 68.98%
![image](https://github.com/user-attachments/assets/51eb94b7-a8e4-427d-b4be-7e36ebca4db2)
This confusion matrix shows strong performance with most predictions correctly placed along the diagonal.
It has fewer misclassifications than Naive Bayes and is comparable or slightly better than SVM.

Step 5: Combining Predictions for Robustness
To build a robust model, we combine the predictions of all three models by taking the mode of their outputs. 
This ensures that even if one model makes an incorrect prediction the final output remains accurate.

Combined Model Accuracy: 60.64%
![image](https://github.com/user-attachments/assets/306af8b2-cd1d-4f89-95cb-8d3f2df422d1)
Each cell shows how many times a true class (rows) was predicted as another class (columns) with high values on the diagonal indicating correct predictions.

Step 6: Creating Prediction Function
Finally, we create a function that takes symptoms as input and predicts the disease using the combined model. The input symptoms are encoded into numerical format and predictions are generated using the trained models.
![image](https://github.com/user-attachments/assets/baaf5c80-eec6-491e-ac47-590e94305cdb)
The output shows predictions from three classifiers where Random Forest predicted "Heart Attack", Naive Bayes Predicted "Urinary tract Infection" and SVM predict "Impetigo". The final combined prediction was "Heart Attack". We can further fine tune this model to make predictions more accurate.





