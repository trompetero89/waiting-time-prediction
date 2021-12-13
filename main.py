# -*- coding: utf-8 -*-
"""

"""

from fastapi import FastAPI, File, Form, UploadFile
import gzip
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO


"""
# loading model
file_path = 'predict_waiting_rf.sav'
pred_model = pickle.load(open(file_path, 'rb'))

"""


app = FastAPI(
    title='Waiting time prediction API',
    description='API that receives parameters to predict waitng time in queue',
    version='0.1'
    
    )

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

# loading model
file_path = 'predict_waiting_rf.sav'
pred_model = pickle.load(open(file_path, 'rb'))

#pred_model = load_zipped_pickle('predict_waiting_rf_zip.sav')


@app.get("/predict-waiting-time")
async def predict_waiting(fecha_inicio:str, tipo_chat:str, tipo_falla:str=None):

    df_prueba = pd.DataFrame({'Fecha_inicio': [fecha_inicio],
                              'Tipo_de_Chat': [tipo_chat],
                              'TipoDeFalla': [tipo_falla],
                       })
    df_prueba['Fecha_inicio'] = pd.to_datetime(df_prueba['Fecha_inicio'])
    df_prueba['Day_conexion'] = pd.DatetimeIndex(df_prueba['Fecha_inicio']).day
    df_prueba['Weeekday_conexion'] = pd.DatetimeIndex(df_prueba['Fecha_inicio']).weekday
    df_prueba['Hora_conexion'] = df_prueba['Fecha_inicio'].dt.hour
    
    df_prueba["Tipo_de_Chat"] = df_prueba["Tipo_de_Chat"].astype('category')
    df_prueba["Tipo_de_Chat_cat"] = df_prueba["Tipo_de_Chat"].cat.codes
    
    
    df_prueba["TipoDeFalla"] = df_prueba["TipoDeFalla"].astype('category')
    df_prueba["TipoDeFalla"] = df_prueba["TipoDeFalla"].replace('Configuración de telefono',
                                                                'Configuración del Teléfono')
    df_prueba["TipoDeFalla"] = df_prueba["TipoDeFalla"].replace('Formato de hora y fecha',
                                                                'Formato de Hora y Fecha')
    df_prueba["TipoDeFalla_cat"] = df_prueba["TipoDeFalla"].cat.codes
    
    features = ['Day_conexion', 'Weeekday_conexion', 'Hora_conexion',
                'Tipo_de_Chat_cat', 'TipoDeFalla_cat']
    df_prueba =df_prueba[features]
    pred_waiting_time = pred_model.predict(df_prueba)[0]
    pred_waiting_time =  np.expm1(pred_waiting_time)
    
    result = {"waiting_time_prediction": pred_waiting_time}
    return result


"""
'2021-08-23 11:47:10.450'
'Reporte Incidencias'
'Configuración de telefono'
"""



"""
@app.post("/predict-waiting-time")
def predict_waiting_time(day_connection:int, weekday_conection:int,
                         hour_connection:int, type_chat:int,
                         type_fail:int):
    
    df_features = pd.DataFrame({'Day_conexion': [day_connection],
                                'Weeekday_conexion': [weekday_conection],
                                'Hora_conexion': [hour_connection],
                                'Tipo_de_Chat_cat': [type_chat],
                                'TipoDeFalla_cat': [type_fail],
                                })
    pred_waiting_time = pred_model.predict(df_features)[0]
    pred_waiting_time =  np.expm1(pred_waiting_time)
    
    result = {"waiting_time_prediction": pred_waiting_time}
    return result
"""
