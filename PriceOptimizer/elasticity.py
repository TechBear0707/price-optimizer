import numpy as np
from scipy.stats import linregress
import pandas as pd

df = pd.read_csv('elastic_input.csv')
df['StayDate'] = pd.to_datetime(df['StayDate'], format='%m/%d/%y')
df['AvgPrice'] = round(df['AvgPrice'], 0)
#df.set_index('StayDate', inplace=True)

# aggregate by ProductID, find top 30 selling products
#top_products = df.groupby('ProductID')['Quantity'].sum().reset_index()
#top_products = top_products.sort_values(by='Quantity', ascending=False)
#top_products = top_products.head(30)

# filter the data to only include the top 30 selling products
#df = df[df['ProductID'].isin(top_products['ProductID'])]

def find_elasticity(df, insert_mean=True):
    """
    Calculate price elasticity of demand for each product

    :param df: DataFrame that contains price per unit and quantity sold
    :param insert_mean: True if mean elasticity should be used for products with insignificant p-values
    :return: Dictionary of product price elasticities
    """

    # calculate aggregate price elasticity of demand for all products
    agg_df = df.groupby('AvgPrice')['Quantity'].sum().reset_index()
    log_x = np.log(agg_df['AvgPrice'])
    log_y = np.log(agg_df['Quantity'])
    agg_slope, agg_intercept, agg_r_value, agg_p_value, agg_std_err = linregress(log_x, log_y)

    # calculate price elasticity of demand for each product and replace with aggregate slope if p-value is insignificant
    if insert_mean:
        elastic_dict = {}
        for product in df['ProductID'].unique():
            prod_df = df.loc[df['ProductID'] == product]
            agg_df = prod_df.groupby('AvgPrice')['Quantity'].sum().reset_index()
            log_x = np.log(agg_df['AvgPrice'])
            log_y = np.log(agg_df['Quantity'])
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
            if p_value < 0.05:
                elastic_dict[product] = abs(slope)
            else:
                elastic_dict[product] = abs(agg_slope)
    # only include products with significant p-values
    else:
        elastic_dict = {}
        for product in df['ProductID'].unique():
            prod_df = df.loc[df['ProductID'] == product]
            agg_df = prod_df.groupby('AvgPrice')['Quantity'].sum().reset_index()
            log_x = np.log(agg_df['AvgPrice'])
            log_y = np.log(agg_df['Quantity'])
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
            if p_value < 0.05:
                elastic_dict[product] = abs(slope)
            else:
                continue

    return elastic_dict


def save_elasticity(df, insert_mean=True):
    """
    Save price elasticity of demand for each product to a CSV file

    :param df: DataFrame that contains price per unit and quantity sold
    :param insert_mean: True if mean elasticity should be used for products with insignificant p-values
    :return: None
    """
    elastic_dict = find_elasticity(df, insert_mean)
    elastic_df = pd.DataFrame(list(elastic_dict.items()), columns=['ProductID', 'Elasticity'])
    elastic_df.to_csv('product_elasticity.csv', index=False)


save_elasticity(df, insert_mean=False)