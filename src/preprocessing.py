

def prepare_data(df):
    # Usuń niepotrzebne dane
    df = df.drop(['Id'], axis=1)

    # łączenie klas (3 -> 4 oraz 8 -> 7)
    df['quality'] = df['quality'].replace({3:5,4:5,8:6,7:6})#({3: 4,8: 7}) # ({3:5,4:5,8:6,7:6})
    print("Rozkład klas (po połączeniu klas):")
    print(df['quality'].value_counts())

    X = df.drop('quality', axis=1)
    y = df['quality']

    return X, y



