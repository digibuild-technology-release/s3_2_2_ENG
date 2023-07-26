import pandas as pd
import glob


def createCsv(input_folder):

    """
    This method creates an input csv considering all the available data in teh folder resources, cleaned, and filtered for 30 minutes slices.
    The final result is called input_df

    input_folder = 'resources/RVENA_23*.csv'
    """

    nomifiles = (glob.glob(input_folder))
    df=pd.DataFrame()

    for nomi in nomifiles:
        A0=pd.read_csv(nomi, sep=';', header=None)
        df=pd.concat([df,A0])

    nomi_originali = df.iloc[:,2].unique() #Vediamo quante grandezze vengono studiate
    # elimina le colonne 'a' e 'b' dal dataframe
    df=df.drop(df.columns[0],axis=1)
    df=df.drop(df.columns[0],axis=1)
    df.columns = ['nome', 'orario', 'valore'] #Cambiamo il nome delle colonne
    df['valore']=df['valore'].str.replace(',', '.') #Aggiustiamo i valori del dataframe 
    df['valore']=df['valore'].str.strip() #Serve per togliere tutti gli spazi da quella colonna
    df['valore']=df['valore'].astype(float) #Rendiamo la colonna dei numeri float
    # Crea un nuovo dataframe con gli orari come prima colonna
    df = df.pivot(index='orario', columns='nome', values='valore')
    df.index=pd.to_datetime(df.index)
    # Reimposta l'indice
    df_30=df.resample('30T').mean()
    #Salva il dataframe
    df_30.to_csv("resources/input_df.csv")

def filterDataset(input_df):

    """
    input_df = "resources/input_df.csv"
    """

    LHV_ng=9.5 #used for conversion from m3/hr to 
    eta_lim=1.40

    dataset=pd.read_csv(input_df)
    dataset['NG Consumption [kW]'] = dataset['CONSUMO GAS (30 minutos)'].diff()*LHV_ng*2
    dataset['eta'] = dataset['ENERGIA INSTANTANEA (15 minuto)']/(dataset['NG Consumption [kW]']+1)
    dataset['Boiler 1 Hours'] = dataset['Horas Funcionamiento Caldera 1 (15 minuto)'].diff()
    dataset['Boiler 2 Hours'] = dataset['Horas Funcionamiento Caldera 2 (15 minuto)'].diff()
    dataset['BH']=dataset['Boiler 1 Hours']+dataset['Boiler 2 Hours']

    #dataset['filter'] = dataset.apply(lambda row: 0 if row['eta'] < eta_lim and row['BH'] > 0 else 1, axis=1) #applica il se

    dataset['filter'] = dataset.apply(lambda row: 0 if row['eta'] < eta_lim else 1, axis=1) #applica il se


    dataset = dataset.loc[dataset['filter'] != 1]

    dataset=dataset.drop('BH', axis=1)
    dataset=dataset.drop('filter', axis=1)

    dataset.fillna(0, inplace=True)

    return dataset