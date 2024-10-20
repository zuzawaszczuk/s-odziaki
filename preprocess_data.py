import pandas as pd

def process_data(path):
    df = pd.read_csv(path)
    df.head()

    df = df[df['unit_sales'] > 10]
    df.drop(["store_nbr", "item_nbr", "id"], axis=1, inplace=True)
    print(df.head())
    df['date'] = pd.to_datetime(df['date'])


    df_grouped = df.groupby('date').agg({'unit_sales': 'sum'}).reset_index()

    df = df_grouped.set_index('date')

    df.index = pd.to_datetime(df.index)
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week


    train = df.loc[df.index < '2017-01-01']
    test = df.loc[df.index >= '2017-01-01']

    FEATURES = ['dayofweek', 'quarter',
        'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    TARGET = 'unit_sales'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    return X_train, y_train, X_test, y_test