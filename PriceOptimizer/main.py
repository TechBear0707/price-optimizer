from lstm import *
from elasticity import *
import random
from helper import *

# read in the data
df = pd.read_csv('forecast_input.csv')
elastic_df = pd.read_csv('elastic_input.csv')
df['StayDate'] = pd.to_datetime(df['StayDate'], format='%m/%d/%y')
df['AvgPrice'] = round(df['AvgPrice'], 0)

# elasticity of each product
elastic_dict = find_elasticity(elastic_df, insert_mean=False)

# filter to products in df that have elastic values
df = df[df['ProductID'].isin(elastic_dict.keys())]

# forecast for all products
forecast_df = lstm_forecast(df, aggregate_by_product=True)

# iterate through the forecast_df and find the optimal price for each product
for index, col in forecast_df.iterrows():
    product_id = col['ProductID']
    ticket_type = col['TicketType']
    avg_price = get_price(df, product_id, ticket_type=None, aggregate_by_product=True)

    # Initial price and quantity
    P0 = avg_price  # Initial price
    Q0 = forecast_df.at[index, 'Quantity']  # Initial quantity demanded

    # Price elasticity of demand
    # add a try except block to handle key errors
    try:
        E = elastic_dict[product_id]  # Elasticity
    except KeyError:
        # exit iteration if key error occurs
        continue

    # create a list with values ranging from 0.5 to 1.5, with 0.05 increments
    max_revenue = 0
    optimal_price = 0
    optimal_quantity = 0
    optimal_revenue = 0
    price_changes = np.arange(-0.5, 0.5, 0.01)
    for price_change in price_changes:
        new_price = P0 * (1 + price_change)
        new_quantity = Q0 * (1 + (-E * price_change))
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
