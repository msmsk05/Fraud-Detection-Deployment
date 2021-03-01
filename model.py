{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "religious-honor",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "pd.options.display.float_format = '${:,.2f}'.format\n",
    "pd.set_option('display.max_rows', 500)\n",
    "pd.set_option('display.max_columns', 500)\n",
    "pd.set_option('display.width', 1000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "veterinary-saudi",
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
       "      <th>Client Id</th>\n",
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>$933,424,371.16</td>\n",
       "      <td>440</td>\n",
       "      <td>$1,447.78</td>\n",
       "      <td>$0.08</td>\n",
       "      <td>$3,113.22</td>\n",
       "      <td>8</td>\n",
       "      <td>$2.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>$937,548,621.16</td>\n",
       "      <td>304</td>\n",
       "      <td>$925.82</td>\n",
       "      <td>$0.18</td>\n",
       "      <td>$4,450.15</td>\n",
       "      <td>13</td>\n",
       "      <td>$2.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>$1,984,910,128.16</td>\n",
       "      <td>267</td>\n",
       "      <td>$951.50</td>\n",
       "      <td>$0.09</td>\n",
       "      <td>$1,544.20</td>\n",
       "      <td>5</td>\n",
       "      <td>$2.22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>$1,516,817,781.16</td>\n",
       "      <td>252</td>\n",
       "      <td>$166.87</td>\n",
       "      <td>$0.75</td>\n",
       "      <td>$180.18</td>\n",
       "      <td>1</td>\n",
       "      <td>$0.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>$987,372,117.16</td>\n",
       "      <td>217</td>\n",
       "      <td>$207.57</td>\n",
       "      <td>$0.38</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>$1.17</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Client Id  Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate\n",
       "0   $933,424,371.16       440              $1,447.78        $0.08 $3,113.22             8                 $2.27\n",
       "1   $937,548,621.16       304                $925.82        $0.18 $4,450.15            13                 $2.05\n",
       "2 $1,984,910,128.16       267                $951.50        $0.09 $1,544.20             5                 $2.22\n",
       "3 $1,516,817,781.16       252                $166.87        $0.75   $180.18             1                 $0.50\n",
       "4   $987,372,117.16       217                $207.57        $0.38     $0.00             0                 $1.17"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1=pd.read_excel(\"Analytics Toutes les données du site Web User Explorer 20191010-20210227 (1).xlsx\", sheet_name=\"Dataset1\")\n",
    "df1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ranking-glossary",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4999 entries, 0 to 4998\n",
      "Data columns (total 7 columns):\n",
      " #   Column                 Non-Null Count  Dtype  \n",
      "---  ------                 --------------  -----  \n",
      " 0   Client Id              4999 non-null   float64\n",
      " 1   Sessions               4999 non-null   int64  \n",
      " 2   Avg. Session Duration  4999 non-null   float64\n",
      " 3   Bounce Rate            4999 non-null   float64\n",
      " 4   Revenue                4999 non-null   float64\n",
      " 5   Transactions           4999 non-null   int64  \n",
      " 6   Goal Conversion Rate   4999 non-null   float64\n",
      "dtypes: float64(5), int64(2)\n",
      "memory usage: 273.5 KB\n"
     ]
    }
   ],
   "source": [
    "df1.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "competent-catering",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1[\"Client Id\"]=df1[\"Client Id\"].astype(\"str\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "after-collect",
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
       "      <th>Client Id</th>\n",
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>933424371.1588883</td>\n",
       "      <td>440</td>\n",
       "      <td>$1,447.78</td>\n",
       "      <td>$0.08</td>\n",
       "      <td>$3,113.22</td>\n",
       "      <td>8</td>\n",
       "      <td>$2.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>937548621.1571213</td>\n",
       "      <td>304</td>\n",
       "      <td>$925.82</td>\n",
       "      <td>$0.18</td>\n",
       "      <td>$4,450.15</td>\n",
       "      <td>13</td>\n",
       "      <td>$2.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1984910128.157773</td>\n",
       "      <td>267</td>\n",
       "      <td>$951.50</td>\n",
       "      <td>$0.09</td>\n",
       "      <td>$1,544.20</td>\n",
       "      <td>5</td>\n",
       "      <td>$2.22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1516817781.1592765</td>\n",
       "      <td>252</td>\n",
       "      <td>$166.87</td>\n",
       "      <td>$0.75</td>\n",
       "      <td>$180.18</td>\n",
       "      <td>1</td>\n",
       "      <td>$0.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>987372117.1567248</td>\n",
       "      <td>217</td>\n",
       "      <td>$207.57</td>\n",
       "      <td>$0.38</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>$1.17</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Client Id  Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate\n",
       "0   933424371.1588883       440              $1,447.78        $0.08 $3,113.22             8                 $2.27\n",
       "1   937548621.1571213       304                $925.82        $0.18 $4,450.15            13                 $2.05\n",
       "2   1984910128.157773       267                $951.50        $0.09 $1,544.20             5                 $2.22\n",
       "3  1516817781.1592765       252                $166.87        $0.75   $180.18             1                 $0.50\n",
       "4   987372117.1567248       217                $207.57        $0.38     $0.00             0                 $1.17"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "instructional-truth",
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
       "      <th>Client Id</th>\n",
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>$1,404,280,423.16</td>\n",
       "      <td>3</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>$1,404,697,019.16</td>\n",
       "      <td>3</td>\n",
       "      <td>$47.33</td>\n",
       "      <td>$0.67</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>$1,405,041,907.16</td>\n",
       "      <td>3</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>$1,405,054,552.16</td>\n",
       "      <td>3</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>$1,405,602,678.16</td>\n",
       "      <td>3</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Client Id  Sessions  Avg. Session Duration  Bounce Rate  Revenue  Transactions  Goal Conversion Rate\n",
       "0 $1,404,280,423.16         3                  $0.00        $1.00        0             0                 $0.00\n",
       "1 $1,404,697,019.16         3                 $47.33        $0.67        0             0                 $1.00\n",
       "2 $1,405,041,907.16         3                  $0.00        $1.00        0             0                 $0.00\n",
       "3 $1,405,054,552.16         3                  $0.00        $1.00        0             0                 $0.00\n",
       "4 $1,405,602,678.16         3                  $0.00        $1.00        0             0                 $0.00"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2=pd.read_excel(\"Analytics Toutes les données du site Web User Explorer 20191010-20210227.xlsx\", sheet_name=\"Dataset1\")\n",
    "df2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "advised-essex",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2[\"Client Id\"]=df2[\"Client Id\"].astype(\"str\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ancient-engineering",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5000 entries, 0 to 4999\n",
      "Data columns (total 7 columns):\n",
      " #   Column                 Non-Null Count  Dtype  \n",
      "---  ------                 --------------  -----  \n",
      " 0   Client Id              5000 non-null   object \n",
      " 1   Sessions               5000 non-null   int64  \n",
      " 2   Avg. Session Duration  5000 non-null   float64\n",
      " 3   Bounce Rate            5000 non-null   float64\n",
      " 4   Revenue                5000 non-null   int64  \n",
      " 5   Transactions           5000 non-null   int64  \n",
      " 6   Goal Conversion Rate   5000 non-null   float64\n",
      "dtypes: float64(3), int64(3), object(1)\n",
      "memory usage: 273.6+ KB\n"
     ]
    }
   ],
   "source": [
    "df2.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "professional-tragedy",
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
       "      <th>Client Id</th>\n",
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>933424371.1588883</td>\n",
       "      <td>440</td>\n",
       "      <td>$1,447.78</td>\n",
       "      <td>$0.08</td>\n",
       "      <td>$3,113.22</td>\n",
       "      <td>8</td>\n",
       "      <td>$2.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>937548621.1571213</td>\n",
       "      <td>304</td>\n",
       "      <td>$925.82</td>\n",
       "      <td>$0.18</td>\n",
       "      <td>$4,450.15</td>\n",
       "      <td>13</td>\n",
       "      <td>$2.05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1984910128.157773</td>\n",
       "      <td>267</td>\n",
       "      <td>$951.50</td>\n",
       "      <td>$0.09</td>\n",
       "      <td>$1,544.20</td>\n",
       "      <td>5</td>\n",
       "      <td>$2.22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1516817781.1592765</td>\n",
       "      <td>252</td>\n",
       "      <td>$166.87</td>\n",
       "      <td>$0.75</td>\n",
       "      <td>$180.18</td>\n",
       "      <td>1</td>\n",
       "      <td>$0.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>987372117.1567248</td>\n",
       "      <td>217</td>\n",
       "      <td>$207.57</td>\n",
       "      <td>$0.38</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>$1.17</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Client Id  Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate\n",
       "0   933424371.1588883       440              $1,447.78        $0.08 $3,113.22             8                 $2.27\n",
       "1   937548621.1571213       304                $925.82        $0.18 $4,450.15            13                 $2.05\n",
       "2   1984910128.157773       267                $951.50        $0.09 $1,544.20             5                 $2.22\n",
       "3  1516817781.1592765       252                $166.87        $0.75   $180.18             1                 $0.50\n",
       "4   987372117.1567248       217                $207.57        $0.38     $0.00             0                 $1.17"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.concat([df1,df2])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "neither-impression",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9999, 7)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "accredited-nightlife",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"bought\"]=df.Transactions.apply(lambda x: 1 if x>=1 else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "aware-input",
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
       "      <th>Client Id</th>\n",
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "      <th>bought</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>933424371.1588883</td>\n",
       "      <td>440</td>\n",
       "      <td>$1,447.78</td>\n",
       "      <td>$0.08</td>\n",
       "      <td>$3,113.22</td>\n",
       "      <td>8</td>\n",
       "      <td>$2.27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>937548621.1571213</td>\n",
       "      <td>304</td>\n",
       "      <td>$925.82</td>\n",
       "      <td>$0.18</td>\n",
       "      <td>$4,450.15</td>\n",
       "      <td>13</td>\n",
       "      <td>$2.05</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1984910128.157773</td>\n",
       "      <td>267</td>\n",
       "      <td>$951.50</td>\n",
       "      <td>$0.09</td>\n",
       "      <td>$1,544.20</td>\n",
       "      <td>5</td>\n",
       "      <td>$2.22</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1516817781.1592765</td>\n",
       "      <td>252</td>\n",
       "      <td>$166.87</td>\n",
       "      <td>$0.75</td>\n",
       "      <td>$180.18</td>\n",
       "      <td>1</td>\n",
       "      <td>$0.50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>987372117.1567248</td>\n",
       "      <td>217</td>\n",
       "      <td>$207.57</td>\n",
       "      <td>$0.38</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>$1.17</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Client Id  Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate  bought\n",
       "0   933424371.1588883       440              $1,447.78        $0.08 $3,113.22             8                 $2.27       1\n",
       "1   937548621.1571213       304                $925.82        $0.18 $4,450.15            13                 $2.05       1\n",
       "2   1984910128.157773       267                $951.50        $0.09 $1,544.20             5                 $2.22       1\n",
       "3  1516817781.1592765       252                $166.87        $0.75   $180.18             1                 $0.50       1\n",
       "4   987372117.1567248       217                $207.57        $0.38     $0.00             0                 $1.17       0"
      ]
     },
     "execution_count": 12,
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
   "execution_count": 13,
   "id": "identified-groove",
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
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>bought</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>$4.29</td>\n",
       "      <td>$26.29</td>\n",
       "      <td>$0.85</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>$43.30</td>\n",
       "      <td>$318.44</td>\n",
       "      <td>$0.32</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Sessions  Avg. Session Duration  Bounce Rate\n",
       "bought                                              \n",
       "0          $4.29                 $26.29        $0.85\n",
       "1         $43.30                $318.44        $0.32"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"bought\").agg({\"Sessions\":\"mean\", \"Avg. Session Duration\":\"mean\", \"Bounce Rate\":\"mean\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "burning-serbia",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "bought\n",
       "0    9890\n",
       "1     109\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"bought\").size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "published-figure",
   "metadata": {},
   "outputs": [],
   "source": [
    "buyers=df[df.bought==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "completed-spell",
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
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "      <th>bought</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>$109.00</td>\n",
       "      <td>$109.00</td>\n",
       "      <td>$109.00</td>\n",
       "      <td>$109.00</td>\n",
       "      <td>$109.00</td>\n",
       "      <td>$109.00</td>\n",
       "      <td>$109.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>$43.30</td>\n",
       "      <td>$318.44</td>\n",
       "      <td>$0.32</td>\n",
       "      <td>$602.77</td>\n",
       "      <td>$1.26</td>\n",
       "      <td>$1.60</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>$65.31</td>\n",
       "      <td>$252.31</td>\n",
       "      <td>$0.17</td>\n",
       "      <td>$687.89</td>\n",
       "      <td>$1.40</td>\n",
       "      <td>$0.48</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>$4.00</td>\n",
       "      <td>$24.20</td>\n",
       "      <td>$0.02</td>\n",
       "      <td>$1.20</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$0.42</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>$13.00</td>\n",
       "      <td>$168.24</td>\n",
       "      <td>$0.19</td>\n",
       "      <td>$184.55</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$1.33</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>$20.00</td>\n",
       "      <td>$231.56</td>\n",
       "      <td>$0.30</td>\n",
       "      <td>$363.08</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$1.57</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>$44.00</td>\n",
       "      <td>$386.61</td>\n",
       "      <td>$0.41</td>\n",
       "      <td>$751.88</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$1.97</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>$440.00</td>\n",
       "      <td>$1,447.78</td>\n",
       "      <td>$0.79</td>\n",
       "      <td>$4,450.15</td>\n",
       "      <td>$13.00</td>\n",
       "      <td>$2.70</td>\n",
       "      <td>$1.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate  bought\n",
       "count   $109.00                $109.00      $109.00   $109.00       $109.00               $109.00 $109.00\n",
       "mean     $43.30                $318.44        $0.32   $602.77         $1.26                 $1.60   $1.00\n",
       "std      $65.31                $252.31        $0.17   $687.89         $1.40                 $0.48   $0.00\n",
       "min       $4.00                 $24.20        $0.02     $1.20         $1.00                 $0.42   $1.00\n",
       "25%      $13.00                $168.24        $0.19   $184.55         $1.00                 $1.33   $1.00\n",
       "50%      $20.00                $231.56        $0.30   $363.08         $1.00                 $1.57   $1.00\n",
       "75%      $44.00                $386.61        $0.41   $751.88         $1.00                 $1.97   $1.00\n",
       "max     $440.00              $1,447.78        $0.79 $4,450.15        $13.00                 $2.70   $1.00"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "buyers.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "impressed-latitude",
   "metadata": {},
   "outputs": [],
   "source": [
    "non_buyers=df[df.bought==0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "working-exemption",
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
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "      <th>bought</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>$9,890.00</td>\n",
       "      <td>$9,890.00</td>\n",
       "      <td>$9,890.00</td>\n",
       "      <td>$9,890.00</td>\n",
       "      <td>$9,890.00</td>\n",
       "      <td>$9,890.00</td>\n",
       "      <td>$9,890.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>$4.29</td>\n",
       "      <td>$26.29</td>\n",
       "      <td>$0.85</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.28</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>$6.37</td>\n",
       "      <td>$83.16</td>\n",
       "      <td>$0.21</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.44</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>$2.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.05</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>$2.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.67</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>$3.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>$4.00</td>\n",
       "      <td>$8.79</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.50</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>$217.00</td>\n",
       "      <td>$1,382.67</td>\n",
       "      <td>$1.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>$2.52</td>\n",
       "      <td>$0.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate    bought\n",
       "count $9,890.00              $9,890.00    $9,890.00 $9,890.00     $9,890.00             $9,890.00 $9,890.00\n",
       "mean      $4.29                 $26.29        $0.85     $0.00         $0.00                 $0.28     $0.00\n",
       "std       $6.37                 $83.16        $0.21     $0.00         $0.00                 $0.44     $0.00\n",
       "min       $2.00                  $0.00        $0.05     $0.00         $0.00                 $0.00     $0.00\n",
       "25%       $2.00                  $0.00        $0.67     $0.00         $0.00                 $0.00     $0.00\n",
       "50%       $3.00                  $0.00        $1.00     $0.00         $0.00                 $0.00     $0.00\n",
       "75%       $4.00                  $8.79        $1.00     $0.00         $0.00                 $0.50     $0.00\n",
       "max     $217.00              $1,382.67        $1.00     $0.00         $0.00                 $2.52     $0.00"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "non_buyers.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "mexican-tattoo",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anaconda belgeler\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='bought'>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 504x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7,6))\n",
    "sns.barplot(df.groupby(\"bought\")[\"Avg. Session Duration\"].mean().index,df.groupby(\"bought\")[\"Avg. Session Duration\"].mean().values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "opposite-material",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anaconda belgeler\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='bought'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 504x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7,6))\n",
    "sns.barplot(df.groupby(\"bought\")[\"Bounce Rate\"].mean().index,df.groupby(\"bought\")[\"Bounce Rate\"].mean().values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "informative-moore",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\Anaconda belgeler\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='bought'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 504x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7,6))\n",
    "sns.barplot(df.groupby(\"bought\")[\"Sessions\"].mean().index,df.groupby(\"bought\")[\"Sessions\"].mean().values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "veterinary-cache",
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
       "      <th>Client Id</th>\n",
       "      <th>Sessions</th>\n",
       "      <th>Avg. Session Duration</th>\n",
       "      <th>Bounce Rate</th>\n",
       "      <th>Revenue</th>\n",
       "      <th>Transactions</th>\n",
       "      <th>Goal Conversion Rate</th>\n",
       "      <th>bought</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>933424371.1588883</td>\n",
       "      <td>440</td>\n",
       "      <td>$1,447.78</td>\n",
       "      <td>$0.08</td>\n",
       "      <td>$3,113.22</td>\n",
       "      <td>8</td>\n",
       "      <td>$2.27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>937548621.1571213</td>\n",
       "      <td>304</td>\n",
       "      <td>$925.82</td>\n",
       "      <td>$0.18</td>\n",
       "      <td>$4,450.15</td>\n",
       "      <td>13</td>\n",
       "      <td>$2.05</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1984910128.157773</td>\n",
       "      <td>267</td>\n",
       "      <td>$951.50</td>\n",
       "      <td>$0.09</td>\n",
       "      <td>$1,544.20</td>\n",
       "      <td>5</td>\n",
       "      <td>$2.22</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1516817781.1592765</td>\n",
       "      <td>252</td>\n",
       "      <td>$166.87</td>\n",
       "      <td>$0.75</td>\n",
       "      <td>$180.18</td>\n",
       "      <td>1</td>\n",
       "      <td>$0.50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>987372117.1567248</td>\n",
       "      <td>217</td>\n",
       "      <td>$207.57</td>\n",
       "      <td>$0.38</td>\n",
       "      <td>$0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>$1.17</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Client Id  Sessions  Avg. Session Duration  Bounce Rate   Revenue  Transactions  Goal Conversion Rate  bought\n",
       "0   933424371.1588883       440              $1,447.78        $0.08 $3,113.22             8                 $2.27       1\n",
       "1   937548621.1571213       304                $925.82        $0.18 $4,450.15            13                 $2.05       1\n",
       "2   1984910128.157773       267                $951.50        $0.09 $1,544.20             5                 $2.22       1\n",
       "3  1516817781.1592765       252                $166.87        $0.75   $180.18             1                 $0.50       1\n",
       "4   987372117.1567248       217                $207.57        $0.38     $0.00             0                 $1.17       0"
      ]
     },
     "execution_count": 22,
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
   "execution_count": 23,
   "id": "considered-pension",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "contemporary-rainbow",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=df.bought"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "educational-hypothetical",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "X=df[[\"Sessions\", \"Avg. Session Duration\", \"Bounce Rate\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "better-wrestling",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "threatened-cologne",
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn import over_sampling\n",
    "from imblearn.over_sampling import SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "printable-creek",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before OverSampling, counts of label '1': 92\n",
      "Before OverSampling, counts of label '0': 7907 \n",
      "\n",
      "After OverSampling, the shape of X_train_new: (15814, 3)\n",
      "After OverSampling, the shape of y_train_new: (15814,) \n",
      "\n",
      "After OverSampling, counts of label '1': 7907\n",
      "After OverSampling, counts of label '0': 7907\n"
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
   "cell_type": "code",
   "execution_count": 29,
   "id": "virtual-dominant",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "thermal-inclusion",
   "metadata": {},
   "outputs": [],
   "source": [
    "rf=RandomForestClassifier(n_estimators=450).fit(X_train_new, y_train_new)\n",
    "y_pred=rf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "criminal-leeds",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "agreed-halifax",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1940   43]\n",
      " [   5   12]]\n"
     ]
    }
   ],
   "source": [
    "print(confusion_matrix(y_test,y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "comparable-freeze",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.98      0.99      1983\n",
      "           1       0.22      0.71      0.33        17\n",
      "\n",
      "    accuracy                           0.98      2000\n",
      "   macro avg       0.61      0.84      0.66      2000\n",
      "weighted avg       0.99      0.98      0.98      2000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "fundamental-armenia",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "logreg=LogisticRegression(solver=\"liblinear\", C=2).fit(X_train_new, y_train_new)\n",
    "y_predlog=logreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "packed-blogger",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1865  118]\n",
      " [   2   15]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.94      0.97      1983\n",
      "           1       0.11      0.88      0.20        17\n",
      "\n",
      "    accuracy                           0.94      2000\n",
      "   macro avg       0.56      0.91      0.58      2000\n",
      "weighted avg       0.99      0.94      0.96      2000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(confusion_matrix(y_test,y_predlog))\n",
    "print(classification_report(y_test,y_predlog))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "interested-saint",
   "metadata": {},
   "outputs": [],
   "source": [
    "log_reg=LogisticRegression(solver=\"liblinear\", C=2).fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "northern-moderator",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "rapid-norfolk",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(log_reg, open( \"model.pkl\", \"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "divided-treatment",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
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
 "nbformat_minor": 5
}
