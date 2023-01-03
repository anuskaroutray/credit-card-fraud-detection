#Install the xgboost module
# !pip install xgboost

# !pip install imbalanced-learn

"""# Import the required Libraries"""

# Import the required libraries for ML algorithms
# import imblearn
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, plot_confusion_matrix

# Import the required libraries for CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

class Args():
	def __init__(self):
	
		self.data_path = "./creditcard.csv"
		self.random_state = 0
		self.test_size = 0.2
		self.batch_size = 2048
		self.epochs = 35

args = Args()

st.title('Credit Card Fraud Detection using Machine Learning and Deep Learning')
uploaded_file = st.file_uploader("Upload the credit card transactions csv file")

# args.data_path = uploaded_file.name

df = pd.read_csv(args.data_path)
st.dataframe(df)

data_vis = st.checkbox('Do you want to explore dataset visualisation')

# Separate the features and labels from the dataframe
X = np.array(df.drop(columns = ["Time", "Amount", "Class"]))
y = np.array(df["Class"])

if data_vis:
	st.set_option('deprecation.showPyplotGlobalUse', False)
	# """# Data visualisation"""
	# y_1 represents number of fraudulent transactions
	y_1 = np.sum(y)

	# y_0 represents number of non-fraudulent transactions
	y_0 = y.shape[0] - y_1
	print(y_1)
	print(y_0)

	# Plot the Class distribution of dataset
	plt.figure(figsize = (7, 5))
	plt.bar([0, 1], [y_0, y_1], width = 0.2)
	plt.xlabel("Class Label")
	plt.ylabel("Frequency")
	plt.title("Class Distribution Plot")
	plt.text(1.2, 251000, "Class Distribution: From the class distribution plot,\nwe can infer the data is indeed skewed with the number\nof non-fraudulent transactions being very high compared\nto fraudulent transactions.")
	# plt.show()
	st.pyplot(plt.show())

# """Class Distribution: From the class distribution plot, we can infer the data is indeed skewed with the number of non-fraudulent transactions being very high compared to fraudulent transactions."""

	# Plot several statistical quantities of the dataset
	plt.figure(figsize = (7, 5))
	non_fraud = np.array(df[df["Class"] == 0].describe()["Amount"].drop(["count", "max"]))
	fraud = np.array(df[df["Class"] == 1].describe()["Amount"].drop(["count", "max"]))
	x_labels = np.array([i for i in range(non_fraud.shape[0])])
	print(x_labels)
	my_xticks = ['Mean','Std','Min','25%', '50%', '75%']
	plt.xticks(x_labels, my_xticks)
	plt.plot(x_labels, non_fraud, c = "r", label = "Non Fraud Case Amount Statistics")
	plt.plot(x_labels, fraud, c = "b", label = "Fraud Case Amount Statistics")
	plt.title("Line Graph")
	plt.text(5.5, 220,"Statistical Quantities: From the above plot,\nwe get an idea about the central tendencies\nof the dataset corresponding to fraudulent\nand non fraudulent transactions.")
	plt.legend()
	st.pyplot(plt.show())
	# plt.show()


# """Statistical Quantities: From the above plot, we get an idea about the central tendencies of the dataset corresponding to fraudulent and non fraudulent transactions.
 #####################################



# No Data Balancing
# data_type = st.selectbox("Select the type of data to be used for fitting/training the model:", ["Balanced", "Unbalanced"])

# x_train, x_test, y_train, y_test = None, None, None, None
# if data_type == "Unbalanced":
# 	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
# elif data_type == "Balanced":
# 	oversample = SMOTE(sampling_strategy = 0.1)
# 	X_balanced, y_balanced = oversample.fit_resample(X, y)
# 	x_train, x_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state = args.random_state, test_size = args.test_size)

# # Function to compute the evaluation metrics given the predicted and true labels
def compute_metrics(preds, true):
	
	accuracy = accuracy_score(true, preds)
	precision = precision_score(true, preds)
	f1 = f1_score(true, preds)
	recall = recall_score(true, preds)
	roc_auc = roc_auc_score(true, preds)
	prc = precision_recall_curve(true, preds)

	return accuracy, precision, recall, f1, roc_auc, prc



ml_model_type = st.selectbox("Select the ML model type:", ["Decision Tree", "K Nearest Neighbours", "Support Vector Machines", 
										"Random Forest Classifier", "Logistic Regression", "XGBoost Classifier"])


# Baseline CNN Class following the architecture mentioned in the paper
class BaselineCNN(nn.Module):
	def __init__(self):
		super(BaselineCNN, self).__init__()

		self.conv1 = nn.Conv1d(1, 32, 2)
		self.bn1 = nn.BatchNorm1d(32)
		self.dropout1 = nn.Dropout(0.2)

		self.conv2 = nn.Conv1d(32, 64, 2)
		self.bn2 = nn.BatchNorm1d(64)
		self.dropout2 = nn.Dropout(0.2)

		self.dense = nn.Linear(1664, 64)
		self.dropout3 = nn.Dropout(0.5)
		self.dense1 = nn.Linear(64, 1)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

  	# Forward propagation given an input
	def forward(self, x):
		x = self.dropout1(self.bn1(self.relu(self.conv1(x))))
		x = self.dropout2(self.bn2(self.relu(self.conv2(x))))
		x = torch.flatten(x, start_dim = 1)
		x = self.relu(self.dense(x))
		x = self.dropout3(x)
		x = self.dense1(x)
		x = self.sigmoid(x)
		return x

# CNN Class with 14 layers following the architecture mentioned in the paper
class CNN14(nn.Module):
	def __init__(self):
		super(CNN14, self).__init__()

		self.conv1 = nn.Conv1d(1, 32, 2)
		self.bn1 = nn.BatchNorm1d(32)
		self.dropout1 = nn.Dropout(0.2)
		
		self.conv2 = nn.Conv1d(32, 64, 2)
		self.bn2 = nn.BatchNorm1d(64)
		self.dropout2 = nn.Dropout(0.5)
		
		self.dense = nn.Linear(1664, 64)
		self.dropout3 = nn.Dropout(0.5)
		self.dense1 = nn.Linear(64, 100)
		self.dense2 = nn.Linear(100, 50)
		self.dense3 = nn.Linear(50, 25)

		self.cls = nn.Linear(25, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		x = self.dropout1(self.bn1(self.relu(self.conv1(x))))
		x = self.dropout2(self.bn2(self.relu(self.conv2(x))))
		x = torch.flatten(x, start_dim = 1)
		x = self.relu(self.dense(x))
		x = self.dropout3(x)
		x = self.relu(self.dense1(x))
		x = self.relu(self.dense2(x))
		x = self.relu(self.dense3(x))
		x = self.sigmoid(self.cls(x))

		return x

# CNN Class with 17 layers following the architecture mentioned in the paper
class CNN17(nn.Module):
	def __init__(self):
		super(CNN17, self).__init__()

		self.conv1 = nn.Conv1d(1, 32, 2)
		self.bn1 = nn.BatchNorm1d(32)
		self.dropout1 = nn.Dropout(0.2)

		self.conv2 = nn.Conv1d(32, 64, 2)
		self.bn2 = nn.BatchNorm1d(64)
		self.dropout2 = nn.Dropout(0.5)

		self.conv3 = nn.Conv1d(64, 64, 2)
		self.bn3 = nn.BatchNorm1d(64)
		self.dropout3 = nn.Dropout(0.25)

		self.dense = nn.Linear(1600, 64)
		self.dropout3 = nn.Dropout(0.5)
		self.dense1 = nn.Linear(64, 100)
		self.dense2 = nn.Linear(100, 50)
		self.dense3 = nn.Linear(50, 25)

		self.cls = nn.Linear(25, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		x = self.dropout1(self.bn1(self.relu(self.conv1(x))))
		x = self.dropout2(self.bn2(self.relu(self.conv2(x))))
		x = self.dropout3(self.bn3(self.relu(self.conv3(x))))
		x = torch.flatten(x, start_dim = 1)
		x = self.relu(self.dense(x))
		x = self.dropout3(x)
		x = self.relu(self.dense1(x))
		x = self.relu(self.dense2(x))
		x = self.relu(self.dense3(x))
		x = self.sigmoid(self.cls(x))

		return x

# CNN Class with 20 layers following the architecture mentioned in the paper
class CNN20(nn.Module):
	def __init__(self):
		super(CNN20, self).__init__()

		self.conv1 = nn.Conv1d(1, 32, 2)
		self.bn1 = nn.BatchNorm1d(32)
		self.dropout1 = nn.Dropout(0.2)

		self.conv2 = nn.Conv1d(32, 64, 2)
		self.bn2 = nn.BatchNorm1d(64)
		self.dropout2 = nn.Dropout(0.5)

		self.conv3 = nn.Conv1d(64, 64, 2)
		self.bn3 = nn.BatchNorm1d(64)
		self.dropout3 = nn.Dropout(0.5)

		self.conv4 = nn.Conv1d(64, 64, 2)
		self.bn4 = nn.BatchNorm1d(64)
		self.dropout4 = nn.Dropout(0.25)

		self.dense = nn.Linear(1536, 64)
		self.dropout3 = nn.Dropout(0.5)
		self.dense1 = nn.Linear(64, 100)
		self.dense2 = nn.Linear(100, 50)
		self.dense3 = nn.Linear(50, 25)

		self.cls = nn.Linear(25, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		x = self.dropout1(self.bn1(self.relu(self.conv1(x))))
		x = self.dropout2(self.bn2(self.relu(self.conv2(x))))
		x = self.dropout3(self.bn3(self.relu(self.conv3(x))))
		x = self.dropout4(self.bn4(self.relu(self.conv4(x))))
		x = torch.flatten(x, start_dim = 1)
		x = self.relu(self.dense(x))
		x = self.dropout3(x)
		x = self.relu(self.dense1(x))
		x = self.relu(self.dense2(x))
		x = self.relu(self.dense3(x))
		x = self.sigmoid(self.cls(x))

		return x


# Set the criterion or loss function
criterion = nn.BCELoss()
device = "cpu"


def train_evaluate(model):

	running_loss = 0.0
	preds = []
	true = []
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# forward propagation
		outputs = model(inputs.unsqueeze(1).float().to(device))
		loss = criterion(outputs.squeeze(), labels.squeeze().float().to(device))

		pred = outputs.detach().cpu().numpy()
		pred[pred >= 0.5] = 1
		pred[pred < 0.5] = 0
		pred = list(pred.reshape(pred.shape[0])) #

		preds.extend(pred)
		true.extend(labels.detach().cpu().numpy())

		running_loss += loss.item()

	running_loss /= i

	preds = np.array(preds)
	true = np.array(true)
	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(preds, true)

	# print('Finished evaluation')
	# print("The training metrics are: ")
	# print(f"\tAccuracy: {train_accuracy: .4f}, Precision: {train_precision: .4f}, Recall: {train_recall: .4f}, F1: {train_f1: .4f}\n")

	return train_accuracy, train_precision, train_recall, train_f1, running_loss


def test_evaluate(model):

	running_loss = 0.0
	preds = []
	true = []
	for i, data in enumerate(testloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# forward propagation
		outputs = model(inputs.unsqueeze(1).float().to(device))
		loss = criterion(outputs.squeeze(), labels.squeeze().float().to(device))

		pred = outputs.detach().cpu().numpy()
		pred[pred >= 0.5] = 1
		pred[pred < 0.5] = 0
		pred = list(pred.reshape(pred.shape[0]))

		preds.extend(pred)
		true.extend(labels.detach().cpu().numpy())

		running_loss += loss.item()

	running_loss /= (i+1)

	preds = np.array(preds)
	true = np.array(true)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(preds, true)

	# print('Finished evaluation')
	# print("The test metrics are: ")
	# print(f"\tAccuracy: {test_accuracy: .4f}, Precision: {test_precision: .4f}, Recall: {test_recall: .4f}, F1: {test_f1: .4f}\n")

	return test_accuracy, test_precision, test_recall, test_f1, running_loss




if ml_model_type == "Decision Tree":
	# # Fit the decision tree classifier and evaluate
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
	# if data_type == "Unbalanced":
	filename = "decision_tree_unbalanced.pkl"
	with open(filename, "rb") as file:
		decision_tree = pickle.load(file)	
	# else:
	# 	filename = "decision_tree_balanced.pkl"
	# 	with open(filename, "rb") as file:
	# 		decision_tree = pickle.load(file)

	train_preds = decision_tree.predict(x_train)
	test_preds = decision_tree.predict(x_test)

	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(train_preds, y_train)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(test_preds, y_test)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif ml_model_type == "K Nearest Neighbours":
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
	# if data_type == "Unbalanced":
	filename = "knn_unbalanced.pkl"
	with open(filename, "rb") as file:
		decision_tree = pickle.load(file)	
	# else:
	# 	filename = "knn_balanced.pkl"
	# 	with open(filename, "rb") as file:
	# 		decision_tree = pickle.load(file)

	train_preds = decision_tree.predict(x_train)
	test_preds = decision_tree.predict(x_test)

	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(train_preds, y_train)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(test_preds, y_test)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif ml_model_type == "Support Vector Machines":
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
	# if data_type == "Unbalanced":
	filename = "svm_unbalanced.pkl"
	with open(filename, "rb") as file:
		decision_tree = pickle.load(file)	
	# else:
	# 	filename = "svm_balanced.pkl"
	# 	with open(filename, "rb") as file:
	# 		decision_tree = pickle.load(file)

	train_preds = decision_tree.predict(x_train)
	test_preds = decision_tree.predict(x_test)

	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(train_preds, y_train)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(test_preds, y_test)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif ml_model_type == "Random Forest Classifier":
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
	# if data_type == "Unbalanced":
	filename = "rfc_unbalanced.pkl"
	with open(filename, "rb") as file:
		decision_tree = pickle.load(file)	
	# else:
	# 	filename = "rfc_balanced.pkl"
	# 	with open(filename, "rb") as file:
	# 		decision_tree = pickle.load(file)

	train_preds = decision_tree.predict(x_train)
	test_preds = decision_tree.predict(x_test)

	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(train_preds, y_train)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(test_preds, y_test)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif ml_model_type == "Logistic Regression":
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
	# if data_type == "Unbalanced":
	filename = "lr_unbalanced.pkl"
	with open(filename, "rb") as file:
		decision_tree = pickle.load(file)	
	# else:
	# 	filename = "lr_balanced.pkl"
	# 	with open(filename, "rb") as file:
	# 		decision_tree = pickle.load(file)

	train_preds = decision_tree.predict(x_train)
	test_preds = decision_tree.predict(x_test)

	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(train_preds, y_train)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(test_preds, y_test)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif ml_model_type == "XGBoost Classifier":
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = args.random_state, test_size = args.test_size)
	# if data_type == "Unbalanced":
	filename = "xgb_unbalanced.pkl"
	with open(filename, "rb") as file:
		decision_tree = pickle.load(file)	
	# else:
	# 	filename = "xgb_balanced.pkl"
	# 	with open(filename, "rb") as file:
	# 		decision_tree = pickle.load(file)

	train_preds = decision_tree.predict(x_train)
	test_preds = decision_tree.predict(x_test)

	train_accuracy, train_precision, train_recall, train_f1, train_roc, train_prc = compute_metrics(train_preds, y_train)
	test_accuracy, test_precision, test_recall, test_f1, test_roc, test_prc = compute_metrics(test_preds, y_test)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)




dl_model_type = st.selectbox("Select the DL model type:", ["Baseline CNN", "CNN with 14 layers", "CNN with 17 layers", "CNN with 20 layers"])

if dl_model_type == "Baseline CNN":

	oversample = SMOTE(sampling_strategy = 0.1)
	X_balanced, y_balanced = oversample.fit_resample(X, y)
	x_train, x_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state = args.random_state, test_size = args.test_size)

	# Prepare the training dataloader
	train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	trainloader = DataLoader(train_dataset, batch_size = args.batch_size)

	# Prepare the test dataloader
	test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	testloader = DataLoader(test_dataset, batch_size = args.batch_size)

	model = BaselineCNN()

	# if data_type == "Balanced":
	model = torch.load("baseline_cnn_balanced.pt")
	# else:
		# model = torch.load("baseline_cnn_unbalanced.pt")

	train_accuracy, train_precision, train_recall, train_f1, running_loss = train_evaluate(model)
	test_accuracy, test_precision, test_recall, test_f1, running_loss = test_evaluate(model)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif dl_model_type == "CNN with 14 layers":
	
	oversample = SMOTE(sampling_strategy = 0.1)
	X_balanced, y_balanced = oversample.fit_resample(X, y)
	x_train, x_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state = args.random_state, test_size = args.test_size)

	# Prepare the training dataloader
	train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	trainloader = DataLoader(train_dataset, batch_size = args.batch_size)

	# Prepare the test dataloader
	test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	testloader = DataLoader(test_dataset, batch_size = args.batch_size)

	model = CNN14()

	# if data_type == "Balanced":
	model = torch.load("cnn14_balanced.pt")
	# else:
		# model = torch.load("cnn14_unbalanced.pt")

	train_accuracy, train_precision, train_recall, train_f1, running_loss = train_evaluate(model)
	test_accuracy, test_precision, test_recall, test_f1, running_loss = test_evaluate(model)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif dl_model_type == "CNN with 17 layers":
	
	oversample = SMOTE(sampling_strategy = 0.1)
	X_balanced, y_balanced = oversample.fit_resample(X, y)
	x_train, x_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state = args.random_state, test_size = args.test_size)

	# Prepare the training dataloader
	train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	trainloader = DataLoader(train_dataset, batch_size = args.batch_size)

	# Prepare the test dataloader
	test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	testloader = DataLoader(test_dataset, batch_size = args.batch_size)
	
	model = CNN17()

	# if data_type == "Balanced":
	model = torch.load("cnn17_balanced.pt")
	# else:
		# model = torch.load("cnn17_unbalanced.pt")

	train_accuracy, train_precision, train_recall, train_f1, running_loss = train_evaluate(model)
	test_accuracy, test_precision, test_recall, test_f1, running_loss = test_evaluate(model)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)

elif dl_model_type == "CNN with 20 layers":
	
	oversample = SMOTE(sampling_strategy = 0.1)
	X_balanced, y_balanced = oversample.fit_resample(X, y)
	x_train, x_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state = args.random_state, test_size = args.test_size)

	# Prepare the training dataloader
	train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	trainloader = DataLoader(train_dataset, batch_size = args.batch_size)

	# Prepare the test dataloader
	test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	testloader = DataLoader(test_dataset, batch_size = args.batch_size)
	
	model = CNN20()

	# if data_type == "Balanced":
	model = torch.load("cnn20_balanced.pt")
	# else:
		# model = torch.load("cnn20_unbalanced.pt")

	train_accuracy, train_precision, train_recall, train_f1, running_loss = train_evaluate(model)
	test_accuracy, test_precision, test_recall, test_f1, running_loss = test_evaluate(model)

	metric_df = pd.DataFrame({"Dataset": ["Train", "Test"], 
								"Accuracy": [train_accuracy, test_accuracy],
								"Precision": [train_precision, test_precision],
								"Recall": [train_recall, test_recall],
								"F1": [train_f1, test_f1]})
	st.dataframe(metric_df)