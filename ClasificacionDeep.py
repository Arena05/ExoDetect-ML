import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Cargar y preparar datos
def load_and_prepare_data():
    print("Cargando datos...")
    
    urls = {
        'koi': "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=koi_disposition,ra,dec,koi_period,koi_duration,koi_depth,koi_prad,koi_insol,koi_teq,koi_steff,koi_slogg,koi_srad&format=csv",
        'toi': "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&select=tfopwg_disp,ra,dec,pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,st_teff,st_logg,st_rad&format=csv",
        'k2': "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+disposition,ra,dec,pl_orbper,pl_trandur,(pl_trandep*10000),pl_rade,pl_insol,pl_eqt,st_teff,st_logg,st_rad+from+k2pandc+order+by+hostname+asc,pl_letter+asc,pl_name+asc&format=csv"
    }

    # Leer datos
    df_koi = pd.read_csv(urls['koi'], low_memory=False)
    df_toi = pd.read_csv(urls['toi'], low_memory=False)
    df_k2 = pd.read_csv(urls['k2'], low_memory=False)

    # Renombrar columnas
    df_koi_renamed = df_koi.rename(columns={
        'koi_disposition': 'disposition',
        'koi_period': 'period',
        'koi_duration': 'duration',
        'koi_depth': 'depth',
        'koi_prad': 'planet_radius',
        'koi_insol': 'insolation',
        'koi_teq': 'equilibrium_temp',
        'koi_steff': 'stellar_teff',
        'koi_slogg': 'stellar_logg',
        'koi_srad': 'stellar_radius'
    })

    df_toi_renamed = df_toi.rename(columns={
        'tfopwg_disp': 'disposition',
        'pl_orbper': 'period',
        'pl_trandurh': 'duration',
        'pl_trandep': 'depth',
        'pl_rade': 'planet_radius',
        'pl_insol': 'insolation',
        'pl_eqt': 'equilibrium_temp',
        'st_teff': 'stellar_teff',
        'st_logg': 'stellar_logg',
        'st_rad': 'stellar_radius'
    })

    df_k2_renamed = df_k2.rename(columns={
        'pl_orbper': 'period',
        'pl_trandur': 'duration',
        '(pl_trandep*10000)': 'depth',
        'pl_rade': 'planet_radius',
        'pl_insol': 'insolation',
        'pl_eqt': 'equilibrium_temp',
        'st_teff': 'stellar_teff',
        'st_logg': 'stellar_logg',
        'st_rad': 'stellar_radius'
    })

    # Combinar datos
    df_combined = pd.concat([df_koi_renamed, df_toi_renamed, df_k2_renamed], 
                           ignore_index=True, sort=False)

    # Limpiar datos
    df_clean = df_combined.dropna().reset_index(drop=True)
    
    # Mapear disposiciones
    disposition_mapping = {
        'APC': 'CANDIDATE',
        'CP': 'CONFIRMED', 
        'FA': 'FALSE POSITIVE',
        'FP': 'FALSE POSITIVE',
        'KP': 'CONFIRMED',
        'PC': 'CANDIDATE',
        'REFUTED': 'FALSE POSITIVE'
    }
    
    df_clean['disposition'] = df_clean['disposition'].replace(disposition_mapping)
    df_clean = df_clean[df_clean['disposition'] != 'CANDIDATE'].reset_index(drop=True)
    
    # Codificar variable objetivo
    le = LabelEncoder()
    df_clean['disposition'] = le.fit_transform(df_clean['disposition'])
    
    return df_clean, le

# Entrenar modelos
def train_models(X_train, y_train, model_type, params):
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=params['random_state']
        )
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['random_state']
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            C=params['C'],
            max_iter=params['max_iter'],
            random_state=params['random_state']
        )
    
    model.fit(X_train, y_train)
    return model

# Interfaz de usuario
def get_user_input():
    print("\n=== CLASIFICADOR DE EXOPLANETAS ===")
    print("Introduce los 11 parámetros del exoplaneta:")
    
    features = [
        'ra', 'dec', 'period', 'duration', 'depth', 
        'planet_radius', 'insolation', 'equilibrium_temp', 
        'stellar_teff', 'stellar_logg', 'stellar_radius'
    ]
    
    user_data = []
    for feature in features:
        value = float(input(f"{feature}: "))
        user_data.append(value)
    
    return np.array([user_data])

def main():
    # Cargar y preparar datos
    df, label_encoder = load_and_prepare_data()
    
    # Preparar características y objetivo
    X = df.drop('disposition', axis=1).values
    y = df['disposition'].values
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Menú de selección de modelo
    print("Selecciona el modelo de clasificación:")
    print("1. Random Forest")
    print("2. Decision Tree")
    print("3. Logistic Regression")
    
    choice = input("Ingresa tu elección (1-3): ")
    
    model_params = {}
    model_name = ""
    
    if choice == "1":
        model_name = "Random Forest"
        print(f"\nConfigurando {model_name}")
        model_params = {
            'n_estimators': int(input("n_estimators (400,800,1200,1600): ")),
            'max_depth': int(input("max_depth (10,20,30,40): ")),
            'min_samples_split': int(input("min_samples_split (5,10,15,20): ")),
            'random_state': int(input("random_state (10,20,30,42): "))
        }
    elif choice == "2":
        model_name = "Decision Tree"
        print(f"\nConfigurando {model_name}")
        model_params = {
            'max_depth': int(input("max_depth (5,8,11,14): ")),
            'min_samples_split': int(input("min_samples_split (5,10,15,20): ")),
            'min_samples_leaf': int(input("min_samples_leaf (2,4,6,8): ")),
            'random_state': int(input("random_state (10,20,30,42): "))
        }
    elif choice == "3":
        model_name = "Logistic Regression"
        print(f"\nConfigurando {model_name}")
        model_params = {
            'C': float(input("C (0.01,0.1,1,10,100): ")),
            'max_iter': int(input("max_iter (100,200,400,600): ")),
            'random_state': int(input("random_state (10,20,30,42): "))
        }
    else:
        print("Elección inválida. Usando Random Forest por defecto.")
        model_name = "Random Forest"
        model_params = {
            'n_estimators': 800,
            'max_depth': 20,
            'min_samples_split': 10,
            'random_state': 42
        }
    
    # Entrenar modelo
    print(f"\nEntrenando modelo {model_name}...")
    model = train_models(X_train_scaled, y_train, model_name, model_params)
    
    # Calcular precisión
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Obtener datos del usuario
    user_input = get_user_input()
    user_input_scaled = scaler.transform(user_input)
    
    # Realizar predicción
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Modelo utilizado: {model_name}")
    print(f"Precisión del modelo: {accuracy:.2%}")
    
    if prediction[0] == 0:
        print("Clasificación: CONFIRMED")
    else:
        print("Clasificación: FALSE POSITIVE")
    
    print(f"Probabilidades: [CONFIRMED: {prediction_proba[0][0]:.2%}, FALSE POSITIVE: {prediction_proba[0][1]:.2%}]")

if __name__ == "__main__":
    main()