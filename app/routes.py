import os
import pandas as pd 
import numpy as np 
import pickle
from flask import Blueprint, render_template, request 

import json 
routes = Blueprint("routes", __name__)

""" RUTAS HOME"""
@routes.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

def prediccionArbol(prediccion):
    pred = np.array(prediccion).reshape(1,11)
    print(pred)
    tree = pickle.load(open('./app/model-winequality-white.pkl','rb'))
    result = tree.predict(pred)
    return result[0]

@routes.route("/clasificar", methods = ["GET",'POST'])
def predict():
 if request.method == 'POST':
     lista_prediccion = request.form.to_dict() #retorna el diccionario de las variables enviadas por post
     lista_prediccion=list(lista_prediccion.values())#convierto los valores del diccionario a una lista
     lista_prediccion = list(map(float, lista_prediccion)) #convierto a flotantes
     print(lista_prediccion)
     resultado = prediccionArbol(lista_prediccion)
     prediccion = str(resultado)
     return render_template("predict.html",prediction=prediccion)
 return render_template("predict.html")