import pandas as pd


# Dataframe: data structure used to store and manipulate tabular data
def create_dataframe():
    data_set = {'id': ['12321421', '2523532', '32523523'], 'age': [22, 55, 100]}
    df = pd.DataFrame(data_set, columns=['id', 'age', ])  # can add more columns, will just init as NaN

    # adding another column (phone) with, must be same length
    df['phone'] = ['111-222-333', '222-333-444', '333-444-555']

    # adding more rows, must be list for 1+ items
    data_set_2 = {'id': ['5555555', '777777'], 'age': [22, 11], 'phone': ['555-555-5555', '111-111-1111']}
    df2 = pd.DataFrame(data_set_2)
    df = df._append(df2, ignore_index=True) # add row at next index of df being appended (0,1,2... 3)

    # adding 1 row via dictionary
    data_set_3 = {'id': '1234567', 'age': 77, 'phone': '000-000-0000'}
    df = df._append(data_set_3, ignore_index=True)

    def createGender(num):
        if num > 50:
            return 'M'
        else:
            return 'F'

    # create gender col using some function logic on data in age col
    df['gender'] = df['age'].apply(createGender)
    print(df)
    print(df.head(1))
    print(df.tail(1))
    # creates a subset with listed cols
    df = df[['id', 'age']]
    # rename specified columns
    df = df.rename(columns={'id': 'UID'})

    # generate statistical summaries of numerical columns
    print(df.describe())
    # get all col data types of df
    print(df.dtypes)
    return df

def read_df():
    df = create_dataframe()
    # get second row of data
    print(df.loc[1])
    data = df.loc[1]
    for key, value in data.items():
        # treat row of data like key value pair, the col = key, value of col = value
        print(key, value)
    # get row 2, value of key 'phone'
    print(df.loc[1]['phone'])

create_dataframe()