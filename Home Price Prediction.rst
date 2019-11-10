.. code:: ipython3

    import os

.. code:: ipython3

    os.getcwd()




.. parsed-literal::

    '/Users/mohammadawais'



.. code:: ipython3

    os.chdir('/Users/mohammadawais/Desktop/ML DATA')

.. code:: ipython3

    os.getcwd()




.. parsed-literal::

    '/Users/mohammadawais/Desktop/ML DATA'



.. code:: ipython3

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn import linear_model
    %matplotlib inline

.. code:: ipython3

    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')

EDA
===

.. code:: ipython3

    train.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Id</th>
          <th>MSSubClass</th>
          <th>MSZoning</th>
          <th>LotFrontage</th>
          <th>LotArea</th>
          <th>Street</th>
          <th>Alley</th>
          <th>LotShape</th>
          <th>LandContour</th>
          <th>Utilities</th>
          <th>...</th>
          <th>PoolArea</th>
          <th>PoolQC</th>
          <th>Fence</th>
          <th>MiscFeature</th>
          <th>MiscVal</th>
          <th>MoSold</th>
          <th>YrSold</th>
          <th>SaleType</th>
          <th>SaleCondition</th>
          <th>SalePrice</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1</td>
          <td>60</td>
          <td>RL</td>
          <td>65.0</td>
          <td>8450</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>Reg</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>2</td>
          <td>2008</td>
          <td>WD</td>
          <td>Normal</td>
          <td>208500</td>
        </tr>
        <tr>
          <td>1</td>
          <td>2</td>
          <td>20</td>
          <td>RL</td>
          <td>80.0</td>
          <td>9600</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>Reg</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>5</td>
          <td>2007</td>
          <td>WD</td>
          <td>Normal</td>
          <td>181500</td>
        </tr>
        <tr>
          <td>2</td>
          <td>3</td>
          <td>60</td>
          <td>RL</td>
          <td>68.0</td>
          <td>11250</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>9</td>
          <td>2008</td>
          <td>WD</td>
          <td>Normal</td>
          <td>223500</td>
        </tr>
        <tr>
          <td>3</td>
          <td>4</td>
          <td>70</td>
          <td>RL</td>
          <td>60.0</td>
          <td>9550</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>2</td>
          <td>2006</td>
          <td>WD</td>
          <td>Abnorml</td>
          <td>140000</td>
        </tr>
        <tr>
          <td>4</td>
          <td>5</td>
          <td>60</td>
          <td>RL</td>
          <td>84.0</td>
          <td>14260</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>12</td>
          <td>2008</td>
          <td>WD</td>
          <td>Normal</td>
          <td>250000</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 81 columns</p>
    </div>



.. code:: ipython3

    train.shape




.. parsed-literal::

    (1460, 81)



.. code:: ipython3

    test.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Id</th>
          <th>MSSubClass</th>
          <th>MSZoning</th>
          <th>LotFrontage</th>
          <th>LotArea</th>
          <th>Street</th>
          <th>Alley</th>
          <th>LotShape</th>
          <th>LandContour</th>
          <th>Utilities</th>
          <th>...</th>
          <th>ScreenPorch</th>
          <th>PoolArea</th>
          <th>PoolQC</th>
          <th>Fence</th>
          <th>MiscFeature</th>
          <th>MiscVal</th>
          <th>MoSold</th>
          <th>YrSold</th>
          <th>SaleType</th>
          <th>SaleCondition</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1461</td>
          <td>20</td>
          <td>RH</td>
          <td>80.0</td>
          <td>11622</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>Reg</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>120</td>
          <td>0</td>
          <td>NaN</td>
          <td>MnPrv</td>
          <td>NaN</td>
          <td>0</td>
          <td>6</td>
          <td>2010</td>
          <td>WD</td>
          <td>Normal</td>
        </tr>
        <tr>
          <td>1</td>
          <td>1462</td>
          <td>20</td>
          <td>RL</td>
          <td>81.0</td>
          <td>14267</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Gar2</td>
          <td>12500</td>
          <td>6</td>
          <td>2010</td>
          <td>WD</td>
          <td>Normal</td>
        </tr>
        <tr>
          <td>2</td>
          <td>1463</td>
          <td>60</td>
          <td>RL</td>
          <td>74.0</td>
          <td>13830</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>MnPrv</td>
          <td>NaN</td>
          <td>0</td>
          <td>3</td>
          <td>2010</td>
          <td>WD</td>
          <td>Normal</td>
        </tr>
        <tr>
          <td>3</td>
          <td>1464</td>
          <td>60</td>
          <td>RL</td>
          <td>78.0</td>
          <td>9978</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>6</td>
          <td>2010</td>
          <td>WD</td>
          <td>Normal</td>
        </tr>
        <tr>
          <td>4</td>
          <td>1465</td>
          <td>120</td>
          <td>RL</td>
          <td>43.0</td>
          <td>5005</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>HLS</td>
          <td>AllPub</td>
          <td>...</td>
          <td>144</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>1</td>
          <td>2010</td>
          <td>WD</td>
          <td>Normal</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 80 columns</p>
    </div>



.. code:: ipython3

    test.shape




.. parsed-literal::

    (1459, 80)



.. code:: ipython3

    plt.style.use(style='ggplot')
    plt.rcParama['figure.figsize']=(10,6)


::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-11-e9e232b4bbfe> in <module>
          1 plt.style.use(style='ggplot')
    ----> 2 plt.rcParama['figure.figsize']=(10,6)
    

    AttributeError: module 'matplotlib.pyplot' has no attribute 'rcParama'


.. code:: ipython3

    train.SalePrice.describe()

.. code:: ipython3

    train.SalePrice.skew()
    # the distribuition is positively skewes meaning the distribuition is on the right mostly




.. parsed-literal::

    1.8828757597682129



.. code:: ipython3

    plt.hist(train.SalePrice,color='blue')
    plt.show()



.. image:: output_14_0.png


.. code:: ipython3

    target=np.log(train.SalePrice)

.. code:: ipython3

    target.skew()
    # now the distribuition becomes more linear




.. parsed-literal::

    0.12133506220520406



.. code:: ipython3

    plt.hist(target,color='red')
    plt.show()



.. image:: output_17_0.png


Feature Engineering
===================

correlation
~~~~~~~~~~~

.. code:: ipython3

    numeric_features=train.select_dtypes(include=[np.number])
    corr=numeric_features.corr()
    corr['SalePrice'].sort_values(ascending=False)[:5]




.. parsed-literal::

    SalePrice      1.000000
    OverallQual    0.790982
    GrLivArea      0.708624
    GarageCars     0.640409
    GarageArea     0.623431
    Name: SalePrice, dtype: float64



outliers remove
~~~~~~~~~~~~~~~

.. code:: ipython3

    plt.scatter(x=train['GarageArea'],y=target)
    plt.ylabel('sale price')
    plt.xlabel('garage area')
    plt.show()



.. image:: output_22_0.png


.. code:: ipython3

    train=train[train['GarageArea']<1200]

.. code:: ipython3

    plt.scatter(x=train['GarageArea'],y=np.log(train.SalePrice))
    plt.xlim(-200,1600)
    plt.ylabel('sale price')
    plt.xlabel('garage area')
    plt.show()



.. image:: output_24_0.png


Null values
~~~~~~~~~~~

.. code:: ipython3

    nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
    nulls.columns=['Null Count']
    nulls.index_name='Feature'

.. code:: ipython3

    nulls




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Null Count</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>PoolQC</td>
          <td>1449</td>
        </tr>
        <tr>
          <td>MiscFeature</td>
          <td>1402</td>
        </tr>
        <tr>
          <td>Alley</td>
          <td>1364</td>
        </tr>
        <tr>
          <td>Fence</td>
          <td>1174</td>
        </tr>
        <tr>
          <td>FireplaceQu</td>
          <td>689</td>
        </tr>
        <tr>
          <td>LotFrontage</td>
          <td>258</td>
        </tr>
        <tr>
          <td>GarageCond</td>
          <td>81</td>
        </tr>
        <tr>
          <td>GarageType</td>
          <td>81</td>
        </tr>
        <tr>
          <td>GarageYrBlt</td>
          <td>81</td>
        </tr>
        <tr>
          <td>GarageFinish</td>
          <td>81</td>
        </tr>
        <tr>
          <td>GarageQual</td>
          <td>81</td>
        </tr>
        <tr>
          <td>BsmtExposure</td>
          <td>38</td>
        </tr>
        <tr>
          <td>BsmtFinType2</td>
          <td>38</td>
        </tr>
        <tr>
          <td>BsmtFinType1</td>
          <td>37</td>
        </tr>
        <tr>
          <td>BsmtCond</td>
          <td>37</td>
        </tr>
        <tr>
          <td>BsmtQual</td>
          <td>37</td>
        </tr>
        <tr>
          <td>MasVnrArea</td>
          <td>8</td>
        </tr>
        <tr>
          <td>MasVnrType</td>
          <td>8</td>
        </tr>
        <tr>
          <td>Electrical</td>
          <td>1</td>
        </tr>
        <tr>
          <td>Utilities</td>
          <td>0</td>
        </tr>
        <tr>
          <td>YearRemodAdd</td>
          <td>0</td>
        </tr>
        <tr>
          <td>MSSubClass</td>
          <td>0</td>
        </tr>
        <tr>
          <td>Foundation</td>
          <td>0</td>
        </tr>
        <tr>
          <td>ExterCond</td>
          <td>0</td>
        </tr>
        <tr>
          <td>ExterQual</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    categorical=train.select_dtypes(exclude=[np.number])
    categorical.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>MSZoning</th>
          <th>Street</th>
          <th>Alley</th>
          <th>LotShape</th>
          <th>LandContour</th>
          <th>Utilities</th>
          <th>LotConfig</th>
          <th>LandSlope</th>
          <th>Neighborhood</th>
          <th>Condition1</th>
          <th>...</th>
          <th>GarageType</th>
          <th>GarageFinish</th>
          <th>GarageQual</th>
          <th>GarageCond</th>
          <th>PavedDrive</th>
          <th>PoolQC</th>
          <th>Fence</th>
          <th>MiscFeature</th>
          <th>SaleType</th>
          <th>SaleCondition</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>count</td>
          <td>1455</td>
          <td>1455</td>
          <td>91</td>
          <td>1455</td>
          <td>1455</td>
          <td>1455</td>
          <td>1455</td>
          <td>1455</td>
          <td>1455</td>
          <td>1455</td>
          <td>...</td>
          <td>1374</td>
          <td>1374</td>
          <td>1374</td>
          <td>1374</td>
          <td>1455</td>
          <td>6</td>
          <td>281</td>
          <td>53</td>
          <td>1455</td>
          <td>1455</td>
        </tr>
        <tr>
          <td>unique</td>
          <td>5</td>
          <td>2</td>
          <td>2</td>
          <td>4</td>
          <td>4</td>
          <td>2</td>
          <td>5</td>
          <td>3</td>
          <td>25</td>
          <td>9</td>
          <td>...</td>
          <td>6</td>
          <td>3</td>
          <td>5</td>
          <td>5</td>
          <td>3</td>
          <td>3</td>
          <td>4</td>
          <td>4</td>
          <td>9</td>
          <td>6</td>
        </tr>
        <tr>
          <td>top</td>
          <td>RL</td>
          <td>Pave</td>
          <td>Grvl</td>
          <td>Reg</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>Inside</td>
          <td>Gtl</td>
          <td>NAmes</td>
          <td>Norm</td>
          <td>...</td>
          <td>Attchd</td>
          <td>Unf</td>
          <td>TA</td>
          <td>TA</td>
          <td>Y</td>
          <td>Ex</td>
          <td>MnPrv</td>
          <td>Shed</td>
          <td>WD</td>
          <td>Normal</td>
        </tr>
        <tr>
          <td>freq</td>
          <td>1147</td>
          <td>1450</td>
          <td>50</td>
          <td>921</td>
          <td>1309</td>
          <td>1454</td>
          <td>1048</td>
          <td>1378</td>
          <td>225</td>
          <td>1257</td>
          <td>...</td>
          <td>867</td>
          <td>605</td>
          <td>1306</td>
          <td>1321</td>
          <td>1335</td>
          <td>2</td>
          <td>157</td>
          <td>48</td>
          <td>1266</td>
          <td>1196</td>
        </tr>
      </tbody>
    </table>
    <p>4 rows × 43 columns</p>
    </div>



.. code:: ipython3

    print('originals')
    print(train.Street.value_counts())


.. parsed-literal::

    originals
    Pave    1450
    Grvl       5
    Name: Street, dtype: int64


.. code:: ipython3

    train['enc_street']=pd.get_dummies(train.Street,drop_first=True)
    test['enc_street']=pd.get_dummies(train.Street,drop_first=True)

.. code:: ipython3

    print('encoded')
    print(train.enc_street.value_counts())


.. parsed-literal::

    encoded
    1    1450
    0       5
    Name: enc_street, dtype: int64


.. code:: ipython3

    condition_pivot=train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)
    condition_pivot.plot(kind='bar',color='blue')
    plt.xlabel('sale condition')
    plt.ylabel('median sale price')
    plt.xticks(rotation=0)
    plt.show()



.. image:: output_32_0.png


missing values
~~~~~~~~~~~~~~

.. code:: ipython3

    data=train.select_dtypes(include=[np.number]).interpolate().dropna()

.. code:: ipython3

    print(sum(data.isnull().sum()!=0))


.. parsed-literal::

    0


no null values in the data

.. code:: ipython3

    y=np.log(train.SalePrice)
    x=data.drop(['SalePrice','Id'],axis=1)

.. code:: ipython3

    y




.. parsed-literal::

    0       12.247694
    1       12.109011
    2       12.317167
    3       11.849398
    4       12.429216
              ...    
    1455    12.072541
    1456    12.254863
    1457    12.493130
    1458    11.864462
    1459    11.901583
    Name: SalePrice, Length: 1455, dtype: float64



.. code:: ipython3

    x




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>MSSubClass</th>
          <th>LotFrontage</th>
          <th>LotArea</th>
          <th>OverallQual</th>
          <th>OverallCond</th>
          <th>YearBuilt</th>
          <th>YearRemodAdd</th>
          <th>MasVnrArea</th>
          <th>BsmtFinSF1</th>
          <th>BsmtFinSF2</th>
          <th>...</th>
          <th>WoodDeckSF</th>
          <th>OpenPorchSF</th>
          <th>EnclosedPorch</th>
          <th>3SsnPorch</th>
          <th>ScreenPorch</th>
          <th>PoolArea</th>
          <th>MiscVal</th>
          <th>MoSold</th>
          <th>YrSold</th>
          <th>enc_street</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>60</td>
          <td>65.0</td>
          <td>8450</td>
          <td>7</td>
          <td>5</td>
          <td>2003</td>
          <td>2003</td>
          <td>196.0</td>
          <td>706</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>61</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2008</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1</td>
          <td>20</td>
          <td>80.0</td>
          <td>9600</td>
          <td>6</td>
          <td>8</td>
          <td>1976</td>
          <td>1976</td>
          <td>0.0</td>
          <td>978</td>
          <td>0</td>
          <td>...</td>
          <td>298</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5</td>
          <td>2007</td>
          <td>1</td>
        </tr>
        <tr>
          <td>2</td>
          <td>60</td>
          <td>68.0</td>
          <td>11250</td>
          <td>7</td>
          <td>5</td>
          <td>2001</td>
          <td>2002</td>
          <td>162.0</td>
          <td>486</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>42</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>9</td>
          <td>2008</td>
          <td>1</td>
        </tr>
        <tr>
          <td>3</td>
          <td>70</td>
          <td>60.0</td>
          <td>9550</td>
          <td>7</td>
          <td>5</td>
          <td>1915</td>
          <td>1970</td>
          <td>0.0</td>
          <td>216</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>35</td>
          <td>272</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2006</td>
          <td>1</td>
        </tr>
        <tr>
          <td>4</td>
          <td>60</td>
          <td>84.0</td>
          <td>14260</td>
          <td>8</td>
          <td>5</td>
          <td>2000</td>
          <td>2000</td>
          <td>350.0</td>
          <td>655</td>
          <td>0</td>
          <td>...</td>
          <td>192</td>
          <td>84</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>12</td>
          <td>2008</td>
          <td>1</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>1455</td>
          <td>60</td>
          <td>62.0</td>
          <td>7917</td>
          <td>6</td>
          <td>5</td>
          <td>1999</td>
          <td>2000</td>
          <td>0.0</td>
          <td>0</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>40</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>8</td>
          <td>2007</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1456</td>
          <td>20</td>
          <td>85.0</td>
          <td>13175</td>
          <td>6</td>
          <td>6</td>
          <td>1978</td>
          <td>1988</td>
          <td>119.0</td>
          <td>790</td>
          <td>163</td>
          <td>...</td>
          <td>349</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2010</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1457</td>
          <td>70</td>
          <td>66.0</td>
          <td>9042</td>
          <td>7</td>
          <td>9</td>
          <td>1941</td>
          <td>2006</td>
          <td>0.0</td>
          <td>275</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>60</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2500</td>
          <td>5</td>
          <td>2010</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1458</td>
          <td>20</td>
          <td>68.0</td>
          <td>9717</td>
          <td>5</td>
          <td>6</td>
          <td>1950</td>
          <td>1996</td>
          <td>0.0</td>
          <td>49</td>
          <td>1029</td>
          <td>...</td>
          <td>366</td>
          <td>0</td>
          <td>112</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>4</td>
          <td>2010</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1459</td>
          <td>20</td>
          <td>75.0</td>
          <td>9937</td>
          <td>5</td>
          <td>6</td>
          <td>1965</td>
          <td>1965</td>
          <td>0.0</td>
          <td>830</td>
          <td>290</td>
          <td>...</td>
          <td>736</td>
          <td>68</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>6</td>
          <td>2008</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    <p>1455 rows × 37 columns</p>
    </div>



.. code:: ipython3

    data




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Id</th>
          <th>MSSubClass</th>
          <th>LotFrontage</th>
          <th>LotArea</th>
          <th>OverallQual</th>
          <th>OverallCond</th>
          <th>YearBuilt</th>
          <th>YearRemodAdd</th>
          <th>MasVnrArea</th>
          <th>BsmtFinSF1</th>
          <th>...</th>
          <th>OpenPorchSF</th>
          <th>EnclosedPorch</th>
          <th>3SsnPorch</th>
          <th>ScreenPorch</th>
          <th>PoolArea</th>
          <th>MiscVal</th>
          <th>MoSold</th>
          <th>YrSold</th>
          <th>SalePrice</th>
          <th>enc_street</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1</td>
          <td>60</td>
          <td>65.0</td>
          <td>8450</td>
          <td>7</td>
          <td>5</td>
          <td>2003</td>
          <td>2003</td>
          <td>196.0</td>
          <td>706</td>
          <td>...</td>
          <td>61</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2008</td>
          <td>208500</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1</td>
          <td>2</td>
          <td>20</td>
          <td>80.0</td>
          <td>9600</td>
          <td>6</td>
          <td>8</td>
          <td>1976</td>
          <td>1976</td>
          <td>0.0</td>
          <td>978</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5</td>
          <td>2007</td>
          <td>181500</td>
          <td>1</td>
        </tr>
        <tr>
          <td>2</td>
          <td>3</td>
          <td>60</td>
          <td>68.0</td>
          <td>11250</td>
          <td>7</td>
          <td>5</td>
          <td>2001</td>
          <td>2002</td>
          <td>162.0</td>
          <td>486</td>
          <td>...</td>
          <td>42</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>9</td>
          <td>2008</td>
          <td>223500</td>
          <td>1</td>
        </tr>
        <tr>
          <td>3</td>
          <td>4</td>
          <td>70</td>
          <td>60.0</td>
          <td>9550</td>
          <td>7</td>
          <td>5</td>
          <td>1915</td>
          <td>1970</td>
          <td>0.0</td>
          <td>216</td>
          <td>...</td>
          <td>35</td>
          <td>272</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2006</td>
          <td>140000</td>
          <td>1</td>
        </tr>
        <tr>
          <td>4</td>
          <td>5</td>
          <td>60</td>
          <td>84.0</td>
          <td>14260</td>
          <td>8</td>
          <td>5</td>
          <td>2000</td>
          <td>2000</td>
          <td>350.0</td>
          <td>655</td>
          <td>...</td>
          <td>84</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>12</td>
          <td>2008</td>
          <td>250000</td>
          <td>1</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>1455</td>
          <td>1456</td>
          <td>60</td>
          <td>62.0</td>
          <td>7917</td>
          <td>6</td>
          <td>5</td>
          <td>1999</td>
          <td>2000</td>
          <td>0.0</td>
          <td>0</td>
          <td>...</td>
          <td>40</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>8</td>
          <td>2007</td>
          <td>175000</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1456</td>
          <td>1457</td>
          <td>20</td>
          <td>85.0</td>
          <td>13175</td>
          <td>6</td>
          <td>6</td>
          <td>1978</td>
          <td>1988</td>
          <td>119.0</td>
          <td>790</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2</td>
          <td>2010</td>
          <td>210000</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1457</td>
          <td>1458</td>
          <td>70</td>
          <td>66.0</td>
          <td>9042</td>
          <td>7</td>
          <td>9</td>
          <td>1941</td>
          <td>2006</td>
          <td>0.0</td>
          <td>275</td>
          <td>...</td>
          <td>60</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2500</td>
          <td>5</td>
          <td>2010</td>
          <td>266500</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1458</td>
          <td>1459</td>
          <td>20</td>
          <td>68.0</td>
          <td>9717</td>
          <td>5</td>
          <td>6</td>
          <td>1950</td>
          <td>1996</td>
          <td>0.0</td>
          <td>49</td>
          <td>...</td>
          <td>0</td>
          <td>112</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>4</td>
          <td>2010</td>
          <td>142125</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1459</td>
          <td>1460</td>
          <td>20</td>
          <td>75.0</td>
          <td>9937</td>
          <td>5</td>
          <td>6</td>
          <td>1965</td>
          <td>1965</td>
          <td>0.0</td>
          <td>830</td>
          <td>...</td>
          <td>68</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>6</td>
          <td>2008</td>
          <td>147500</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    <p>1455 rows × 39 columns</p>
    </div>



splitting dataset
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=.3)

.. code:: ipython3

    x_train.shape




.. parsed-literal::

    (1018, 37)



.. code:: ipython3

    y_train.shape




.. parsed-literal::

    (1018,)



.. code:: ipython3

    x_test.shape




.. parsed-literal::

    (437, 37)



.. code:: ipython3

    y_test.shape




.. parsed-literal::

    (437,)



model building
~~~~~~~~~~~~~~

.. code:: ipython3

    lr=linear_model.LinearRegression()

.. code:: ipython3

    model=lr.fit(x_train,y_train)

R^2 is the measure of how close the data are to the fitted regression
line. in general higher the value of r-squared better will be model fit

R sqaure
~~~~~~~~

.. code:: ipython3

    print('R square value  =',model.score(x_test,y_test))


.. parsed-literal::

    R square value  = 0.8875183697245669


prediction
~~~~~~~~~~

.. code:: ipython3

    predictions=model.predict(x_test)

.. code:: ipython3

    predictions




.. parsed-literal::

    array([11.90469035, 12.06107637, 11.77509768, 11.8100632 , 11.25866937,
           11.35559265, 12.34926065, 11.70425796, 12.02905349, 11.87446969,
           11.80801501, 12.52686871, 12.25245648, 12.67941307, 11.92305837,
           11.61081732, 12.20137161, 11.61639239, 12.38554849, 12.28815813,
           11.33383765, 12.36537881, 11.44292966, 11.71848952, 12.20228959,
           11.92235963, 11.63062159, 11.48132807, 12.33713705, 11.96266175,
           11.65164304, 12.1379272 , 11.65757645, 11.55193673, 11.73448382,
           12.4035182 , 12.6110777 , 12.02337374, 12.3418443 , 11.41588999,
           11.71738256, 12.3911132 , 11.85065155, 11.89540322, 12.60015484,
           12.24454442, 12.12957703, 11.7951424 , 11.8095127 , 12.180554  ,
           11.66523183, 11.42609546, 12.24824924, 11.71241729, 12.10019685,
           12.42252043, 12.26558841, 12.3212882 , 11.62288078, 12.00683815,
           11.2929781 , 11.74735612, 12.58188563, 12.59359153, 12.34360442,
           12.26186741, 12.50320089, 12.73312261, 12.33283042, 11.96893455,
           11.74171086, 11.86452527, 12.03423674, 11.51160908, 11.49350715,
           11.61314217, 12.43615874, 11.80908503, 11.73628853, 11.70593382,
           12.04722408, 12.62967378, 12.30381776, 11.47299612, 11.68599922,
           11.68752062, 12.52628169, 12.47675925, 11.65137047, 11.98262682,
           11.97950668, 11.74139348, 11.76387527, 12.51220185, 11.65291512,
           11.62747812, 12.01942426, 12.08633401, 12.20838078, 12.46773453,
           12.18940914, 11.70665396, 12.6170189 , 11.83557094, 12.21922513,
           11.89502281, 12.31360375, 12.25183751, 11.6756127 , 11.68641542,
           12.08692056, 11.72029377, 11.79963076, 12.32891525, 11.76492117,
           11.63085183, 12.69177756, 11.79403538, 11.83366764, 12.72352098,
           11.69748309, 11.89836707, 12.38490373, 12.3430559 , 11.52721856,
           11.98953723, 12.82334048, 11.62142579, 11.87596281, 11.61840304,
           12.48096193, 12.09120073, 12.14248609, 11.64300394, 11.78283629,
           11.94449145, 12.1134974 , 11.74567565, 11.97650435, 11.32940142,
           12.75475922, 11.67491883, 12.0389726 , 11.60235111, 13.21454617,
           12.77527814, 12.16637656, 11.62541436, 12.19093488, 11.69070197,
           11.79060381, 11.88156531, 11.38882474, 11.9656009 , 11.57937186,
           11.7017095 , 11.6150232 , 11.84821917, 12.936626  , 11.90585246,
           12.18933465, 11.76558834, 11.43052546, 11.98124661, 11.60813196,
           12.3821846 , 12.32370183, 12.66512655, 11.7873107 , 12.70273132,
           12.83952223, 11.38211732, 12.41488813, 12.02207117, 12.42318206,
           11.74740729, 11.6788451 , 12.07661963, 11.82433316, 11.52124676,
           12.38258201, 11.95951187, 11.80216002, 12.65468502, 12.19966111,
           11.86307059, 11.72601683, 11.69277256, 12.60702421, 11.70270835,
           11.5916413 , 12.58879733, 12.63454227, 11.60918657, 11.57970469,
           11.6019075 , 12.00083818, 11.35877465, 12.24338261, 11.7879631 ,
           12.24352666, 12.14653747, 12.7690025 , 12.50296195, 11.86883963,
           11.62318789, 11.6661367 , 11.51401323, 11.70280977, 11.96513069,
           12.4666496 , 11.66986623, 11.56757978, 12.28271524, 12.28362421,
           12.43528184, 12.20331861, 11.58512562, 12.19810662, 11.80219296,
           11.91835106, 12.61964277, 12.18834895, 11.96090128, 11.88198149,
           11.94214137, 12.1587105 , 12.17143433, 12.27045093, 11.74684936,
           11.64857288, 12.05053359, 11.90879387, 12.28727222, 11.50860275,
           11.79413363, 12.41282999, 12.95178398, 12.39528559, 11.63341156,
           11.61779984, 12.50335996, 12.34509827, 12.88099743, 12.37750549,
           12.1881645 , 12.58721831, 11.68748027, 11.94837585, 11.46200506,
           12.20920469, 11.70874386, 12.07999858, 11.63752535, 12.0865961 ,
           11.84563113, 11.72671733, 12.62698004, 11.93249834, 11.37796457,
           11.57586419, 11.701352  , 11.74559236, 12.30489135, 12.60836082,
           12.29970172, 12.02932336, 12.44183778, 12.04539784, 12.09206979,
           12.91122845, 11.24263422, 12.14207017, 11.47462729, 11.466551  ,
           11.98056354, 12.17933162, 12.20869903, 12.16130761, 12.20214185,
           11.72007989, 11.4920951 , 12.14872656, 11.63697815, 11.76659485,
           11.91036141, 12.05745395, 12.80735603, 11.98332709, 11.35748661,
           12.21015576, 11.84437725, 11.79689351, 12.12385453, 11.46887929,
           12.19881172, 11.40319911, 11.87861835, 12.06698065, 12.82218997,
           12.17441385, 12.09289155, 11.73761769, 12.11984571, 12.16380692,
           12.13561625, 12.21023605, 12.2341632 , 12.28837587, 12.37610107,
           12.15803615, 12.29090474, 11.92001372, 11.54973699, 11.32311305,
           12.31217992, 11.36970367, 11.98128157, 12.12598922, 11.32711284,
           12.11695592, 11.78929   , 12.37074252, 11.4376266 , 12.14839505,
           11.66890108, 12.7032954 , 12.79712222, 12.00691575, 12.74433305,
           11.96322387, 12.11804794, 11.78921428, 11.79950657, 11.23655531,
           11.68896699, 11.30531966, 11.43489334, 11.8035444 , 12.14130976,
           11.7113498 , 11.80195956, 12.10925442, 12.35844323, 12.76424945,
           11.70425605, 11.66120034, 11.46201604, 11.73958417, 12.24864242,
           11.71740628, 11.96756773, 11.70633951, 11.95271493, 11.59545762,
           12.68591503, 11.56407609, 11.86507504, 12.23731065, 11.9154616 ,
           11.82224549, 11.83545215, 11.69310807, 11.95660468, 12.30594184,
           12.96819056, 11.67588685, 12.04559747, 11.84404697, 12.71335033,
           11.70049228, 12.88445178, 11.26656666, 11.78772486, 11.78058491,
           12.23819829, 12.46516266, 12.35192164, 11.63070755, 12.40202092,
           12.16170489, 11.61267487, 11.70393225, 12.09654841, 11.92721042,
           12.19707485, 12.54278015, 12.15297165, 11.57278955, 11.47115355,
           11.81047854, 12.05746419, 11.84593748, 11.75827613, 11.33050286,
           11.76621142, 12.04212933, 11.77082987, 11.96323523, 12.38881345,
           11.85761665, 11.94930832, 12.39651275, 11.48534717, 12.10751176,
           11.67826796, 12.50427255, 11.7356985 , 11.60919129, 12.25832871,
           11.58466437, 12.61003475, 12.59774369, 11.90646965, 12.18368494,
           11.59379485, 12.35139454, 11.70982748, 12.26201575, 11.84215051,
           12.24944448, 12.07410701, 12.5232165 , 11.58144275, 11.73352609,
           11.62760322, 11.74771711, 12.04742084, 12.11047093, 11.92775882,
           12.50731459, 11.77893531, 11.70548552, 11.96196402, 11.65530966,
           11.85175393, 12.15610443])



root_mean_squared error
~~~~~~~~~~~~~~~~~~~~~~~

prediction error RMSE calculate the distance between the predicted
values and the actual values

.. code:: ipython3

    print('RMSE. =',mean_squared_error(y_test,predictions))


.. parsed-literal::

    RMSE. = 0.017391505338428974


.. code:: ipython3

    actual_values=y_test
    plt.scatter(predictions,actual_values,alpha=.75,color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Linear Regression Model')
    plt.show()



.. image:: output_58_0.png


improve model
~~~~~~~~~~~~~

ridge regularization: is a process which shrinks the regression
coefficient of less important features

.. code:: ipython3

    for i in range(-2,3):
        alpha=10**i
        rm=linear_model.Ridge(alpha=alpha)
        ridge_model=rm.fit(x_train,y_train)
        preds_ridge=ridge_model.predict(x_test)
        plt.scatter(preds_ridge,actual_values,alpha=.75,color='b')
        plt.xlabel('predicted price')
        plt.ylabel('actual price')
        plt.title('ridge regularisation with alpha {}'.format(alpha))
        overlay='R^2 is: {} \n RMSE is: {}'.format(ridge_model.score(x_test,y_test),mean_squared_error(y_test,preds_ridge))
        plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
        plt.show()



.. image:: output_60_0.png



.. image:: output_60_1.png



.. image:: output_60_2.png



.. image:: output_60_3.png



.. image:: output_60_4.png


has no effect after ridge regularization and also using different values
for the alpha

visualizing result and submitting the result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    submission=pd.DataFrame()

.. code:: ipython3

    submission['Id']=test.Id

.. code:: ipython3

    submission




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1461</td>
        </tr>
        <tr>
          <td>1</td>
          <td>1462</td>
        </tr>
        <tr>
          <td>2</td>
          <td>1463</td>
        </tr>
        <tr>
          <td>3</td>
          <td>1464</td>
        </tr>
        <tr>
          <td>4</td>
          <td>1465</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>1454</td>
          <td>2915</td>
        </tr>
        <tr>
          <td>1455</td>
          <td>2916</td>
        </tr>
        <tr>
          <td>1456</td>
          <td>2917</td>
        </tr>
        <tr>
          <td>1457</td>
          <td>2918</td>
        </tr>
        <tr>
          <td>1458</td>
          <td>2919</td>
        </tr>
      </tbody>
    </table>
    <p>1459 rows × 1 columns</p>
    </div>



.. code:: ipython3

    feats=test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()

.. code:: ipython3

    predictions=model.predict(feats)

.. code:: ipython3

    final_predictions=np.exp(predictions)
    # we did log transformation so to avoid log transformation in the final result we uses the exponential

.. code:: ipython3

    print('original predictions: ',predictions[:10])
    print('final predictions: ',final_predictions[:10])


.. parsed-literal::

    original predictions:  [11.75771357 11.69565273 12.07596932 12.20781588 12.11574211 12.06014422
     12.15684038 12.02388889 12.1702747  11.65415821]
    final predictions:  [127735.06264756 120048.69448084 175600.9417667  200348.95054051
     182725.82928824 172843.91055635 190391.99935082 166689.63446566
     192967.04382081 115169.26696929]


.. code:: ipython3

    submission['SalePrice']=final_predictions

.. code:: ipython3

    submission.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Id</th>
          <th>SalePrice</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1461</td>
          <td>127735.062648</td>
        </tr>
        <tr>
          <td>1</td>
          <td>1462</td>
          <td>120048.694481</td>
        </tr>
        <tr>
          <td>2</td>
          <td>1463</td>
          <td>175600.941767</td>
        </tr>
        <tr>
          <td>3</td>
          <td>1464</td>
          <td>200348.950541</td>
        </tr>
        <tr>
          <td>4</td>
          <td>1465</td>
          <td>182725.829288</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    submission.to_csv('submission1.csv',index=False)

Finish
======
