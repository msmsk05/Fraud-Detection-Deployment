{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-24P_wByMuHX"
   },
   "source": [
    "# WELCOME!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ow9AD-4vMuHX"
   },
   "source": [
    "Welcome to \"***Fraud Detection Project***\". This is the last project of the Capstone Series.\n",
    "\n",
    "One of the challenges in this project is the absence of domain knowledge. So without knowing what the column names are, you will only be interested in their values. The other one is the class frequencies of the target variable are quite imbalanced.\n",
    "\n",
    "You will implement ***Logistic Regression, Random Forest, Neural Network*** algorithms and ***SMOTE*** technique. Also visualize performances of the models using ***Seaborn, Matplotlib*** and ***Yellowbrick*** in a variety of ways.\n",
    "\n",
    "At the end of the project, you will have the opportunity to deploy your model by ***Flask API***.\n",
    "\n",
    "Before diving into the project, please take a look at the Determines and Tasks.\n",
    "\n",
    "- ***NOTE:*** *This tutorial assumes that you already know the basics of coding in Python and are familiar with model deployement (flask api) as well as the theory behind Logistic Regression, Random Forest, Neural Network.*\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dqbMkIZ-MuHY"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "spCFDhO7MuHY"
   },
   "source": [
    "# #Determines\n",
    "The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where it has **492 frauds** out of **284,807** transactions. The dataset is **highly unbalanced**, the positive class (frauds) account for 0.172% of all transactions.\n",
    "\n",
    "**Feature Information:**\n",
    "\n",
    "**Time**: This feature is contains the seconds elapsed between each transaction and the first transaction in the dataset. \n",
    "\n",
    "**Amount**:  This feature is the transaction Amount, can be used for example-dependant cost-senstive learning. \n",
    "\n",
    "**Class**: This feature is the target variable and it takes value 1 in case of fraud and 0 otherwise.\n",
    "\n",
    "---\n",
    "\n",
    "The aim of this project is to predict whether a credit card transaction is fraudulent. Of course, this is not easy to do.\n",
    "First of all, you need to analyze and recognize your data well in order to draw your roadmap and choose the correct arguments you will use. Accordingly, you can examine the frequency distributions of variables. You can observe variable correlations and want to explore multicollinearity. You can show the distribution of the target variable's classes over other variables. \n",
    "Also, it is useful to take missing values and outliers.\n",
    "\n",
    "After these procedures, you can move on to the model building stage by doing the basic data pre-processing you are familiar with. \n",
    "\n",
    "Start with Logistic Regression and evaluate model performance. You will apply the SMOTE technique used to increase the sample for unbalanced data. Next, rebuild your Logistic Regression model with SMOTE applied data to observe its effect.\n",
    "\n",
    "Then, you will use three different algorithms in the model building phase. You have applied Logistic Regression and Random Forest in your previous projects. However, the Deep Learning Neural Network algorithm will appear for the first time.\n",
    "\n",
    "In the final step, you will deploy your model using ***Flask API***."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YOl6z9mXMuHY"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1o6X3hLLMuHZ"
   },
   "source": [
    "# #Tasks\n",
    "\n",
    "#### 1. Exploratory Data Analysis & Data Cleaning\n",
    "\n",
    "- Import Modules, Load Data & Data Review\n",
    "- Exploratory Data Analysis\n",
    "- Data Cleaning\n",
    "\n",
    "\n",
    "\n",
    "    \n",
    "#### 2. Data Preprocessing\n",
    "\n",
    "- Scaling\n",
    "- Train - Test Split\n",
    "\n",
    "\n",
    "#### 3. Model Building\n",
    "\n",
    "- Logistic Regression without SMOTE\n",
    "- Apply SMOTE\n",
    "- Logistic Regression with SMOTE\n",
    "- Random Forest Classifier with SMOTE\n",
    "- Neural Network\n",
    "\n",
    "#### 4. Model Deployement\n",
    "\n",
    "- Save and Export the Model as .pkl\n",
    "- Save and Export Variables as .pkl \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9sDSWJywMuHZ"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fbFMU3AdMuHZ"
   },
   "source": [
    "## 1. Exploratory Data Analysis & Data Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5nmI08_GMuHZ"
   },
   "source": [
    "### Import Modules, Load Data & Data Review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "yKZtJybfMuHa"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import zipfile"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "zf = zipfile.ZipFile('creditcard.zip')\n",
    "df = pd.read_csv(zf.open('creditcard.csv'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>149.62</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>378.66</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>123.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>69.99</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9  ...       V21       V22       V23       V24       V25  \\\n",
       "0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   \n",
       "1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   \n",
       "2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   \n",
       "3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   \n",
       "4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   \n",
       "\n",
       "        V26       V27       V28  Amount  Class  \n",
       "0 -0.189115  0.133558 -0.021053  149.62      0  \n",
       "1  0.125895 -0.008983  0.014724    2.69      0  \n",
       "2 -0.139097 -0.055353 -0.059752  378.66      0  \n",
       "3 -0.221929  0.062723  0.061458  123.50      0  \n",
       "4  0.502292  0.219422  0.215153   69.99      0  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 284807 entries, 0 to 284806\n",
      "Data columns (total 31 columns):\n",
      " #   Column  Non-Null Count   Dtype  \n",
      "---  ------  --------------   -----  \n",
      " 0   Time    284807 non-null  float64\n",
      " 1   V1      284807 non-null  float64\n",
      " 2   V2      284807 non-null  float64\n",
      " 3   V3      284807 non-null  float64\n",
      " 4   V4      284807 non-null  float64\n",
      " 5   V5      284807 non-null  float64\n",
      " 6   V6      284807 non-null  float64\n",
      " 7   V7      284807 non-null  float64\n",
      " 8   V8      284807 non-null  float64\n",
      " 9   V9      284807 non-null  float64\n",
      " 10  V10     284807 non-null  float64\n",
      " 11  V11     284807 non-null  float64\n",
      " 12  V12     284807 non-null  float64\n",
      " 13  V13     284807 non-null  float64\n",
      " 14  V14     284807 non-null  float64\n",
      " 15  V15     284807 non-null  float64\n",
      " 16  V16     284807 non-null  float64\n",
      " 17  V17     284807 non-null  float64\n",
      " 18  V18     284807 non-null  float64\n",
      " 19  V19     284807 non-null  float64\n",
      " 20  V20     284807 non-null  float64\n",
      " 21  V21     284807 non-null  float64\n",
      " 22  V22     284807 non-null  float64\n",
      " 23  V23     284807 non-null  float64\n",
      " 24  V24     284807 non-null  float64\n",
      " 25  V25     284807 non-null  float64\n",
      " 26  V26     284807 non-null  float64\n",
      " 27  V27     284807 non-null  float64\n",
      " 28  V28     284807 non-null  float64\n",
      " 29  Amount  284807 non-null  float64\n",
      " 30  Class   284807 non-null  int64  \n",
      "dtypes: float64(30), int64(1)\n",
      "memory usage: 67.4 MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K22reBkbMuHa"
   },
   "source": [
    "### Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NGyEoz9fJQ0E"
   },
   "source": [
    "### Data Cleaning\n",
    "Check Missing Values and Outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "BvpEPuGAMuHa"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    284315\n",
       "1       492\n",
       "Name: Class, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Class.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anaconda belgeler\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Class', ylabel='count'>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(df.Class)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tMOO7g-sMuHb"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Yf6VvH6WMuHb"
   },
   "source": [
    "## 2. Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OV28RJBeMuHb"
   },
   "source": [
    "#### Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler=MinMaxScaler().fit(df[[\"Amount\"]])\n",
    "df[[\"Amount\"]]=scaler.transform(df[[\"Amount\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>0.005824</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>0.000105</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "\n",
       "         V8        V9  ...       V21       V22       V23       V24       V25  \\\n",
       "0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   \n",
       "1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   \n",
       "\n",
       "        V26       V27       V28    Amount  Class  \n",
       "0 -0.189115  0.133558 -0.021053  0.005824      0  \n",
       "1  0.125895 -0.008983  0.014724  0.000105      0  \n",
       "\n",
       "[2 rows x 31 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hlm6gCsKMuHb"
   },
   "source": [
    "#### Train - Test Split\n",
    "\n",
    "As in this case, for extremely imbalanced datasets you may want to make sure that classes are balanced across train and test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "AuzpxEmKMuHb"
   },
   "outputs": [],
   "source": [
    "X=df.drop(\"Class\", axis=1)\n",
    "y=df.Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn import over_sampling\n",
    "from imblearn.over_sampling import SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before OverSampling, counts of label '1': 344\n",
      "Before OverSampling, counts of label '0': 199020 \n",
      "\n",
      "After OverSampling, the shape of X_train_new: (398040, 30)\n",
      "After OverSampling, the shape of y_train_new: (398040,) \n",
      "\n",
      "After OverSampling, counts of label '1': 199020\n",
      "After OverSampling, counts of label '0': 199020\n"
     ]
    }
   ],
   "source": [
    "print(\"Before OverSampling, counts of label '1': {}\".format(sum(y_train==1)))\n",
    "print(\"Before OverSampling, counts of label '0': {} \\n\".format(sum(y_train==0)))\n",
    "\n",
    "sm = SMOTE(random_state=41)\n",
    "X_train_new, y_train_new = sm.fit_sample(X_train, y_train)\n",
    "\n",
    "print('After OverSampling, the shape of X_train_new: {}'.format(X_train_new.shape))\n",
    "print('After OverSampling, the shape of y_train_new: {} \\n'.format(y_train_new.shape))\n",
    "\n",
    "print(\"After OverSampling, counts of label '1': {}\".format(sum(y_train_new==1)))\n",
    "print(\"After OverSampling, counts of label '0': {}\".format(sum(y_train_new==0)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HO4HAIofMuHc"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MwQdl4PdJQ0I"
   },
   "source": [
    "## 3. Model Building\n",
    "It was previously stated that you need to make class prediction with three different algorithms. As in this case, different approaches are required to obtain better performance on unbalanced data.\n",
    "\n",
    "This dataset is severely **unbalanced** (most of the transactions are non-fraud). So the algorithms are much more likely to classify new observations to the majority class and high accuracy won't tell us anything. To address the problem of imbalanced dataset we can use undersampling and oversampling data approach techniques. Oversampling increases the number of minority class members in the training set. The advantage of oversampling is that no information from the original training set is lost unlike in undersampling, as all observations from the minority and majority classes are kept. On the other hand, it is prone to overfitting. \n",
    "\n",
    "There is a type of oversampling called **[SMOTE](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/)** (Synthetic Minority Oversampling Technique), which we are going to use to make our dataset balanced. It creates synthetic points from the minority class.\n",
    "\n",
    "- It is important that you can evaluate the effectiveness of SMOTE. For this reason, implement the Logistic Regression algorithm in two different ways, with SMOTE applied and without.\n",
    "\n",
    "***Note***: \n",
    "\n",
    "- *Do not forget to import the necessary libraries and modules before starting the model building!*\n",
    "\n",
    "- *If you are going to use the cross validation method to be more sure of the performance of your model for unbalanced data, you should make sure that the class distributions in the iterations are equal. For this case, you should use **[StratifiedKFold](https://www.analyseup.com/python-machine-learning/stratified-kfold.html)** instead of regular cross validation method.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zKZcwgucJQ0I"
   },
   "source": [
    "### Logistic Regression without SMOTE\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows: \n",
    "\n",
    "   i. Import Libraries\n",
    "   \n",
    "   *ii. Model Training*\n",
    "   \n",
    "   *iii. Prediction and Model Evaluating*\n",
    "   \n",
    "   *iv. Plot Precision and Recall Curve*\n",
    "   \n",
    "   *v. Apply and Plot StratifiedKFold*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "o48s5BCdMuHd"
   },
   "source": [
    "***i. Import Libraries***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "3G3cx-UjMuHd"
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6KD76bc5MuHd"
   },
   "source": [
    "***ii. Model Training***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "g7GAK-u3MuHd"
   },
   "outputs": [],
   "source": [
    "lr=LogisticRegression(solver='lbfgs').fit(X_train_new, y_train_new)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uvKAJVTNMuHd"
   },
   "source": [
    "***iii. Prediction and Model Evaluating***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "Kb68hH1TMuHd"
   },
   "outputs": [],
   "source": [
    "y_predlr=lr.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "l193OP5fMuHd"
   },
   "source": [
    "\n",
    "You're evaluating \"accuracy score\"? Is your performance metric reflect real success? You may need to use different metrics to evaluate performance on unbalanced data. You should use **[precision and recall metrics](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate.)**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fUDt5voIMuHe"
   },
   "source": [
    "***iv. Plot Precision and Recall Curve***\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "WI0OI9SDMuHe"
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix, classification_report,precision_recall_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "8ugUuOhhMuHe"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# predict probabilities\n",
    "yhat = lr.predict_proba(X_test)\n",
    "# retrieve just the probabilities for the positive class\n",
    "pos_probs = yhat[:, 1]\n",
    "# plot no skill roc curve\n",
    "plt.figure(figsize=(12,8))\n",
    "plt.plot([0, 1], [0, 1], linestyle='--', label='No Fraud')\n",
    "# calculate roc curve for model\n",
    "fpr, tpr, _ = roc_curve(y_test, pos_probs)\n",
    "# plot model roc curve\n",
    "plt.plot(fpr, tpr, marker='.', label='Fraud')\n",
    "# axis labels\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "# show the legend\n",
    "plt.legend()\n",
    "# show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9300327321870203\n",
      "--------------\n",
      "0.8783783783783784\n",
      "--------------\n",
      "0.0768321513002364\n",
      "--------------\n",
      "0.9815081399295437\n",
      "--------------\n",
      "[[83733  1562]\n",
      " [   18   130]]\n",
      "--------------\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.98      0.99     85295\n",
      "           1       0.08      0.88      0.14       148\n",
      "\n",
      "    accuracy                           0.98     85443\n",
      "   macro avg       0.54      0.93      0.57     85443\n",
      "weighted avg       1.00      0.98      0.99     85443\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(roc_auc_score(y_test, y_predlr))\n",
    "print(\"--------------\")\n",
    "print(recall_score(y_test, y_predlr))\n",
    "print(\"--------------\")\n",
    "print(precision_score(y_test, y_predlr))\n",
    "print(\"--------------\")\n",
    "print(accuracy_score(y_test, y_predlr))\n",
    "print(\"--------------\")\n",
    "print(confusion_matrix(y_test, y_predlr))\n",
    "print(\"--------------\")\n",
    "print(classification_report(y_test, y_predlr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([0.00173215, 0.07683215, 1.        ]),\n",
       " array([1.        , 0.87837838, 0.        ]),\n",
       " array([0, 1], dtype=int64))"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_recall_curve(y_test, y_predlr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "lr_precision, lr_recall, _ = precision_recall_curve(y_test, pos_probs)\n",
    "plt.figure(figsize=(12,8))\n",
    "plt.plot([0, 1], linestyle='--')\n",
    "plt.plot(lr_recall, lr_precision, marker='.')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_3zm70O7JQ0Z"
   },
   "source": [
    "### Random Forest Classifier with SMOTE\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows:\n",
    "\n",
    "   *i. Model Training*\n",
    "   \n",
    "   *ii. Prediction and Model Evaluating*\n",
    "   \n",
    "   *iii. Plot Precision and Recall Curve*\n",
    "   \n",
    "   *iv. Apply and Plot StratifiedKFold*\n",
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sr5U80HbMuHg"
   },
   "source": [
    "***i. Model Training and Prediction***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "kuvRr7f3MuHh"
   },
   "outputs": [],
   "source": [
    "rf=RandomForestClassifier().fit(X_train_new, y_train_new)\n",
    "y_pred=rf.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8bdqEhrdMuHh"
   },
   "source": [
    "***ii. Plot Precision and Recall Curve***\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "smne1OBWMuHh"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# predict probabilities\n",
    "yhatrf = rf.predict_proba(X_test)\n",
    "# retrieve just the probabilities for the positive class\n",
    "pos_probs = yhatrf[:, 1]\n",
    "# plot no skill roc curve\n",
    "plt.plot([0, 1], [0, 1], linestyle='--', label='No Fraud')\n",
    "# calculate roc curve for model\n",
    "fpr, tpr, _ = roc_curve(y_test, pos_probs)\n",
    "# plot model roc curve\n",
    "plt.plot(fpr, tpr, marker='.', label='Fraud')\n",
    "# axis labels\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "# show the legend\n",
    "plt.legend()\n",
    "# show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "WukW9Gb3MuHh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.90191564886887\n",
      "--------------\n",
      "0.8040540540540541\n",
      "--------------\n",
      "0.8623188405797102\n",
      "--------------\n",
      "0.9994382219725431\n",
      "--------------\n",
      "[[85276    19]\n",
      " [   29   119]]\n",
      "--------------\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     85295\n",
      "           1       0.86      0.80      0.83       148\n",
      "\n",
      "    accuracy                           1.00     85443\n",
      "   macro avg       0.93      0.90      0.92     85443\n",
      "weighted avg       1.00      1.00      1.00     85443\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(roc_auc_score(y_test, y_pred))\n",
    "print(\"--------------\")\n",
    "print(recall_score(y_test, y_pred))\n",
    "print(\"--------------\")\n",
    "print(precision_score(y_test, y_pred))\n",
    "print(\"--------------\")\n",
    "print(accuracy_score(y_test, y_pred))\n",
    "print(\"--------------\")\n",
    "print(confusion_matrix(y_test, y_pred))\n",
    "print(\"--------------\")\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = RandomForestClassifier(n_estimators=300, max_depth=16).fit(X_train_new, y_train_new)\n",
    "\n",
    "y_predmodel=model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9119804399041166\n",
      "--------------\n",
      "0.8243243243243243\n",
      "--------------\n",
      "0.7973856209150327\n",
      "--------------\n",
      "0.9993328885923949\n",
      "--------------\n",
      "[[85264    31]\n",
      " [   26   122]]\n",
      "--------------\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     85295\n",
      "           1       0.80      0.82      0.81       148\n",
      "\n",
      "    accuracy                           1.00     85443\n",
      "   macro avg       0.90      0.91      0.91     85443\n",
      "weighted avg       1.00      1.00      1.00     85443\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(roc_auc_score(y_test, y_predmodel))\n",
    "print(\"--------------\")\n",
    "print(recall_score(y_test, y_predmodel))\n",
    "print(\"--------------\")\n",
    "print(precision_score(y_test, y_predmodel))\n",
    "print(\"--------------\")\n",
    "print(accuracy_score(y_test, y_predmodel))\n",
    "print(\"--------------\")\n",
    "print(confusion_matrix(y_test, y_predmodel))\n",
    "print(\"--------------\")\n",
    "print(classification_report(y_test, y_predmodel))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "importances=pd.DataFrame(model.feature_importances_, index=X_train.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>V14</th>\n",
       "      <td>0.187374</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>V10</th>\n",
       "      <td>0.148980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>V4</th>\n",
       "      <td>0.126261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>V12</th>\n",
       "      <td>0.098457</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>V17</th>\n",
       "      <td>0.090573</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>V11</th>\n",
       "      <td>0.071392</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            0\n",
       "V14  0.187374\n",
       "V10  0.148980\n",
       "V4   0.126261\n",
       "V12  0.098457\n",
       "V17  0.090573\n",
       "V11  0.071392"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "importances.sort_values(by=0, ascending=False).head(6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_final=X_train_new[[\"V14\", \"V10\", \"V4\", \"V12\", \"V17\", \"V11\"]]\n",
    "X_test_final=X_test[[\"V14\", \"V10\", \"V4\", \"V12\", \"V17\", \"V11\"]]\n",
    "y_train_final=y_train_new"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "my_model = RandomForestClassifier(n_estimators=350).fit(X_train_final, y_train_final)\n",
    "\n",
    "y_predfinal=my_model.predict(X_test_final)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.831081081081081\n",
      "--------------\n",
      "[[85199    96]\n",
      " [   25   123]]\n",
      "--------------\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     85295\n",
      "           1       0.56      0.83      0.67       148\n",
      "\n",
      "    accuracy                           1.00     85443\n",
      "   macro avg       0.78      0.91      0.83     85443\n",
      "weighted avg       1.00      1.00      1.00     85443\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(recall_score(y_test, y_predfinal))\n",
    "print(\"--------------\")\n",
    "print(confusion_matrix(y_test, y_predfinal))\n",
    "print(\"--------------\")\n",
    "print(classification_report(y_test, y_predfinal))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ife6NlFRJQ0f"
   },
   "source": [
    "### Neural Network\n",
    "\n",
    "In the final step, you will make classification with Neural Network which is a Deep Learning algorithm. \n",
    "\n",
    "Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. They are used in a variety of applications in financial services, from forecasting and marketing research to fraud detection and risk assessment.\n",
    "\n",
    "A neural network contains layers of interconnected nodes. Each node is a perceptron and is similar to a multiple linear regression. The perceptron feeds the signal produced by a multiple linear regression into an activation function that may be nonlinear.\n",
    "\n",
    "In a multi-layered perceptron (MLP), perceptrons are arranged in interconnected layers. The input layer collects input patterns. The output layer has classifications or output signals to which input patterns may map. \n",
    "\n",
    "Hidden layers fine-tune the input weightings until the neural network’s margin of error is minimal. It is hypothesized that hidden layers extrapolate salient features in the input data that have predictive power regarding the outputs.\n",
    "\n",
    "You will discover **[how to create](https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5)** your deep learning neural network model in Python using **[Keras](https://keras.io/about/)**. Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models.\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows:\n",
    "\n",
    "   *i. Import Libraries*\n",
    "   \n",
    "   *ii. Define Model*\n",
    "    \n",
    "   *iii. Compile Model*\n",
    "   \n",
    "   *iv. Fit Model*\n",
    "   \n",
    "   *v. Prediction and Model Evaluating*\n",
    "   \n",
    "   *vi. Plot Precision and Recall Curve*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "InMeP9kgMuHj"
   },
   "source": [
    "***Prediction and Model Evaluating***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "executionInfo": {
     "elapsed": 1897,
     "status": "ok",
     "timestamp": 1610977899555,
     "user": {
      "displayName": "Owen l",
      "photoUrl": "",
      "userId": "01085249422681493006"
     },
     "user_tz": -180
    },
    "id": "LhEc3K9KMuHi"
   },
   "outputs": [],
   "source": [
    "from sklearn.neural_network import MLPClassifier\n",
    "clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train_new, y_train_new)\n",
    "y_predmlp=clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_JAEDNkjMuHj"
   },
   "source": [
    "***Plot Precision and Recall Curve***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# predict probabilities\n",
    "yhatmlp = clf.predict_proba(X_test)\n",
    "# retrieve just the probabilities for the positive class\n",
    "pos_probs = yhatmlp[:, 1]\n",
    "# plot no skill roc curve\n",
    "plt.plot([0, 1], [0, 1], linestyle='--', label='No Fraud')\n",
    "# calculate roc curve for model\n",
    "fpr, tpr, _ = roc_curve(y_test, pos_probs)\n",
    "# plot model roc curve\n",
    "plt.plot(fpr, tpr, marker='.', label='Fraud')\n",
    "# axis labels\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "# show the legend\n",
    "plt.legend()\n",
    "# show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9366216295432547\n"
     ]
    }
   ],
   "source": [
    "print(roc_auc_score(y_test, y_predmlp))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8783783783783784\n",
      "--------------\n",
      "0.22887323943661972\n",
      "--------------\n",
      "0.9946631087391594\n",
      "--------------\n",
      "[[84857   438]\n",
      " [   18   130]]\n",
      "--------------\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.99      1.00     85295\n",
      "           1       0.23      0.88      0.36       148\n",
      "\n",
      "    accuracy                           0.99     85443\n",
      "   macro avg       0.61      0.94      0.68     85443\n",
      "weighted avg       1.00      0.99      1.00     85443\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(recall_score(y_test, y_predmlp))\n",
    "print(\"--------------\")\n",
    "print(precision_score(y_test, y_predmlp))\n",
    "print(\"--------------\")\n",
    "print(accuracy_score(y_test, y_predmlp))\n",
    "print(\"--------------\")\n",
    "print(confusion_matrix(y_test, y_predmlp))\n",
    "print(\"--------------\")\n",
    "print(classification_report(y_test, y_predmlp))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LpbiGnpIxVK3"
   },
   "source": [
    "## 4. Model Deployement\n",
    "You cooked the food in the kitchen and moved on to the serving stage. The question is how do you showcase your work to others? Model Deployement helps you showcase your work to the world and make better decisions with it. But, deploying a model can get a little tricky at times. Before deploying the model, many things such as data storage, preprocessing, model building and monitoring need to be studied.\n",
    "\n",
    "Deployment of machine learning models, means making your models available to your other business systems. By deploying models, other systems can send data to them and get their predictions, which are in turn populated back into the company systems. Through machine learning model deployment, can begin to take full advantage of the model you built.\n",
    "\n",
    "Data science is concerned with how to build machine learning models, which algorithm is more predictive, how to design features, and what variables to use to make the models more accurate. However, how these models are actually used is often neglected. And yet this is the most important step in the machine learning pipline. Only when a model is fully integrated with the business systems, real values ​​can be extract from its predictions.\n",
    "\n",
    "After doing the following operations in this notebook, jump to *Pycharm* and create your web app with Flask API."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oCAYcMLEH_7P"
   },
   "source": [
    "### Save and Export the Model as .pkl\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "id": "MqluJ9yvIOex"
   },
   "outputs": [],
   "source": [
    "import pickle\n",
    "pickle.dump(my_model, open( \"model.pkl\", \"wb\" ))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vaZP1N93IPQi"
   },
   "source": [
    "### Save and Export Variables as .pkl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "q_vA-dJWxfFH"
   },
   "outputs": [],
   "source": [
    "pickle.dump(scaler, open( \"scaler\", \"wb\" ))\n",
    "scaler = pickle.load(open( \"scaler\", \"rb\" ))"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "OV28RJBeMuHb",
    "hlm6gCsKMuHb",
    "zKZcwgucJQ0I",
    "4f8q5y12MuHe",
    "9wvBCEvpJQ0U",
    "_3zm70O7JQ0Z"
   ],
   "name": "Fraud Detection_Student_V2.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
