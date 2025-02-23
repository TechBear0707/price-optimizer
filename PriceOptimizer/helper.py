import pandas as pd
import numpy as np

def get_price(df, product_id, ticket_type, aggregate_by_product):
    '''
    Returns the average price of a product from the past seven days

    :param df: DataFrame that contains price per unit and quantity sold
    :param product_id: The unique identifier for a product
    :param ticket_type: The type of ticket for a product
    :param aggregate_by_product: Boolean flag to indicate if forecasting should be at the Product level only.
    :return: avg_price: The average price of a product from the past seven days
    '''
    if aggregate_by_product:
        sub_df = df.loc[df['ProductID'] == product_id]
        sub_df = sub_df.tail(7)
        avg_price = sub_df['AvgPrice'].mean()
    else:
        sub_df = df.loc[df['ProductID'] == product_id]
        sub_df = sub_df.loc[sub_df['TicketType'] == ticket_type]
        sub_df = sub_df.tail(7)
        avg_price = sub_df['AvgPrice'].mean()

    return avg_price