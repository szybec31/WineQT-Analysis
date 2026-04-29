

def prepare_data(df,mode):
    # Usuń niepotrzebne dane
    df = df.drop(['Id'], axis=1)

    # łączenie klas (3 -> 4 oraz 8 -> 7)
    if mode == "binary":
        df['quality'] = df['quality'].replace({3:5,4:5,8:6,7:6}) # łączenie klas do klas 5 i 6
    elif mode == "4multiclass":
        df['quality'] = df['quality'].replace({3: 4,8: 7})  # łączenie klas (3 -> 4 oraz 8 -> 7)

    print("Rozkład klas (po połączeniu klas):")
    print(df['quality'].value_counts())

    X = df.drop('quality', axis=1)
    y = df['quality']

    return X, y



