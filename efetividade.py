import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.models import load_model

model = load_model('rede.h5')

def welcome():
	return 'Welcome All'

def predict(resultados, produtos_e_servicos, contexto, planejamento, insumos, processos):
	
 
	previsao = model.predict([[resultados,produtos_e_servicos,contexto,planejamento,insumos,processos]])
	print(previsao)
	return previsao

def main():
	st.title('Algoritmo de Previsão')
	html_temp = """
	<div style='background-color:tomato;padding:10px'>
	<h2 style='color:white;text-align:center;'>SAMGe </h2>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)
	resultados = st.number_input('Indicador Resultados (valor entre 0 e 1):')
	produtos_e_servicos = st.number_input('Indicador Produtos e Serviços (valor entre 0 e 1):')
	contexto = st.number_input('Indicador Contexto (valor entre 0 e 1):')
	planejamento = st.number_input('Indicador Planejamento *valor entre 0 e 1):')
	insumos = st.number_input('Indicador Insumos (valor entre 0 e 1):')
	processos = st.number_input('Indicador Processos (valor entre 0 e 1):')
	result = ""
	if st.button('Prever'):
		result = predict(resultados, produtos_e_servicos, contexto, planejamento, insumos, processos) * 100
	st.success('O índice de efetividade previsto é de: {}'.format(result))
	if st.button('Algoritmo e Métricas'):
		st.text('Framework: Tensorflow/Keras')
		st.text('MSE: 0.001112652280116978')
		st.text('RMSE: 0.03335644285766961')
	if st.button('Sobre'):
		st.text('Elaborado por DMAG')

if __name__=='__main__':
	main()


