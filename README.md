# AI_FinalProject

<h3>Hypertension Prediction Project:-</h3>

This project aims to predict the likelihood of hypertension (high blood pressure) based on various health parameters using machine learning techniques. It involves data preprocessing, exploratory data analysis, feature engineering, model building, model selection, and evaluation.

<h3>Project Structure:-</h3>

The project consists of the following files:
<li>hypertension_data.csv: The dataset containing health parameters and the target variable (hypertension status).</li>
<li>hypertension_prediction.ipynb: Jupyter Notebook containing the Python code for the project.</li>
<li>best_model.hdf5: Saved model weights of the best-performing model.</li>

<h3>Project Flow:-</h3>

<h3>Loading Python Libraries:</h3> The necessary Python libraries are imported to support data analysis, visualization, and model building.
<h3>Loading the Dataset:</h3> The hypertension data is loaded into a Pandas DataFrame for further analysis.
<h3>Gaining Insights from Data:</h3> Data exploration is performed to gain insights into the dataset, including descriptive statistics, value counts, and distributions.
<h3>Data Normalization:</h3> The data is normalized using z-score normalization to ensure all values are within a similar range.
<h3>Feature Engineering:</h3> Numeric columns are selected for feature engineering to convert them into numerical features.
<h3>Data Normalization (continued):</h3> After feature engineering, the data is re-normalized.
<h3>Output Column Distribution:</h3> The distribution of the output column (hypertension status) is analyzed and visualized.
<h3>Model Building and Overfitting:</h3> Multiple models with different architectures are built and trained to observe overfitting behavior.
<h3>Phase 3:</h3> Model Selection & Evaluation:</h3> The data is shuffled, split into training and validation sets, and a final model is selected based on performance metrics such as accuracy, loss, and validation metrics.
<h3>Model Evaluation:</h3> The selected model is evaluated using precision, recall, and F1 score metrics. Additionally, the Receiver Operating Characteristic (ROC) curve is plotted to assess the model's performance.
<h3>Conclusion:</h3> The project concludes with a summary of the findings and insights from the analysis.

<h3>Usage:-</h3>

To run the project, follow these steps:
Install the required Python libraries mentioned in the hypertension_prediction.ipynb file.
Place the hypertension_data.csv file in the same directory as the Jupyter Notebook.
Run the cells in the Jupyter Notebook one by one to execute the code and observe the results.
The final selected model's performance and evaluation metrics will be displayed at the end.

Note: The Jupyter Notebook contains detailed explanations and comments for each step to facilitate understanding and customization.

<h3>Dependencies:-</h3>

The project has the following dependencies:
Python (3.x)
Pandas
NumPy
Seaborn
Missingno
Matplotlib
TensorFlow (Keras)
Make sure to have these dependencies installed before running the project.

<h3>Conclusion:-</h3>
This project demonstrates the application of machine learning techniques to predict hypertension based on health parameters. It provides a starting point for further analysis and improvement of the predictive model. The findings and insights gained from this project can contribute to better understanding and management of hypertension in the healthcare domain.
