from lstm import *
from elasticity import *
import random

# read in the data
df = pd.read_csv('Show_Sales.csv')
df['StayDate'] = pd.to_datetime(df['StayDate'], format='%m/%d/%y')
df['AvgPrice'] = round(df['AvgPrice'], 0)
df.set_index('StayDate', inplace=True)

# elasticity of each product
elastic_dict = find_elasticity(df)
print(elastic_dict)

# product dict has product id as key, ticket types as values
prod_dict = {}
for product in df['ProductID'].unique():
    sub_df = df[df['ProductID'] == product]
    tickets = sub_df['TicketType'].unique()
    prod_dict[product] = tickets

# 50 random products
prod_dict = dict(random.sample(prod_dict.items(), 50))

# forecast for all products
forecast_df = lstm_forecast(df, prod_dict)

def get_price(df, product_id, ticket_type):
    '''
    Returns the average price of a product from the past seven days

    :param df: DataFrame that contains price per unit and quantity sold
    :param product_id: The unique identifier for a product
    :param ticket_type: The type of ticket for a product
    :return: avg_price: The average price of a product from the past seven days
    '''
    sub_df = df.loc[df['ProductID'] == product_id]
    sub_df = sub_df.loc[sub_df['TicketType'] == ticket_type]
    sub_df = sub_df.tail(7)
    avg_price = sub_df['AvgPrice'].mean()

    return avg_price


# iterate through the forecast_df and find the optimal price for each product
for index, col in forecast_df.iterrows():
    product_id = col['ProductID']
    ticket_type = col['TicketType']
    avg_price = get_price(df, product_id, ticket_type)

    # Initial price and quantity
    P0 = avg_price  # Initial price
    Q0 = forecast_df.at[index, 'Quantity']  # Initial quantity demanded

    # Price elasticity of demand
    E = elastic_dict[product_id]  # Elasticity

    # create a list with values ranging from 0.5 to 1.5, with 0.05 increments
    max_revenue = 0
    optimal_price = 0
    optimal_quantity = 0
    optimal_revenue = 0
    price_changes = np.arange(-0.5, 0.5, 0.01)
    for price_change in price_changes:
        new_price = P0 * (1 + price_change)
        new_quantity = Q0 * (1 + (E * price_change))
        revenue = new_price * new_quantity
        if revenue > max_revenue:
            max_revenue = revenue
            optimal_price = new_price
            optimal_quantity = new_quantity
            optimal_revenue = revenue

    # insert the avg price and optimal price into the forecast_df
    forecast_df.at[index, 'Elasticity'] = E
    forecast_df.at[index, 'AvgPrice'] = avg_price
    forecast_df.at[index, 'ProjectedRevenue'] = forecast_df.at[index, 'Quantity'] * avg_price
    forecast_df.at[index, 'OptimalPricePercentChange'] = (optimal_price - avg_price) / avg_price

    # round optimal price percent change to 2 decimal places
    forecast_df.at[index, 'OptimalPricePercentChange'] = round(forecast_df.at[index, 'OptimalPricePercentChange'], 2)
    forecast_df.at[index, 'OptimalPrice'] = optimal_price
    forecast_df.at[index, 'OptimalQuantity'] = optimal_quantity
    forecast_df.at[index, 'OptimalRevenue'] = optimal_revenue
    forecast_df.at[index, 'RevenueGain'] = optimal_revenue - forecast_df.at[index, 'ProjectedRevenue']


forecast_df.to_csv('optimal_prices.csv', index=False)
