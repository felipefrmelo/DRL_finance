# Módulo com a execução do Agente

import os
import sys
import click
import logging
import coloredlogs

import numpy as np
import keras.backend as K

from tqdm import tqdm
from time import clock

from agente import Agente
from utils import get_state, get_stock_data, format_currency, format_position

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


@click.command()
@click.option(
    '-es',
    '--executa-stock',
    type=click.Path(exists=True),
    default='data/GOOGL_2018.csv',
    help='Dataset para executar o modelo'
)
@click.option(
    '-ws',
    '--window-size',
    default=10,
    help='Tamanho da janela n-dias dos estados anteriores para normalização'
)
@click.option(
    '-mn',
    '--model-name',
    default='model_GOOGL',
    help='Nome doo modelo para carregar'
)
@click.option(
    '-d',
    '--debug',
    is_flag=True,
    help='Flag para o modo debug'
)

def main(executa_stock, window_size, model_name, debug):
    """ Executa o Stock Asset DSA Bot """
    switch_k_backend_device()
    data = get_stock_data(executa_stock)
    initial_offset = data[1] - data[0]

    if model_name is not None:
        ''' Usa um único modelo '''
        agente = Agente(window_size, pretrained = True, model_name = model_name)
        profit, _ = evaluate_model(agente, data, window_size, debug)
        show_eval_result(model_name, profit, initial_offset)
        del agente
    else:
        ''' Trabalha com vários modelos '''
        for model in os.listdir('models'):
            if not os.path.isdir('models/{}'.format(model)):
                agente = Agente(window_size, pretrained = True, model_name = model)
                profit = evaluate_model(agente, data, window_size, debug)
                show_eval_result(model, profit, initial_offset)
                del agente


def evaluate_model(agente, data, window_size, debug):
    data_length = len(data) - 1
    state = get_state(data, 0, window_size + 1)
    total_profit = 0
    agente.inventory = []
    history = []
    
    for t in range(data_length):
        action = agente.act(state, is_eval=True)
        
        # Esperar
        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0
        
        # Comprar
        if action == 1:
            agente.inventory.append(data[t])
            history.append((data[t], 'Compra'))
            if debug:
                logging.debug('Comprar por: {}'.format(format_currency(data[t])))
        
        # Vender
        elif action == 2 and len(agente.inventory) > 0:
            history.append((data[t], 'Venda'))
            bought_price = agente.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            if debug:
                logging.debug('Vender por: {} | Posição: {}'.format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        else:
            history.append((data[t], 'Aguardar'))

        done = True if t == data_length - 1 else False
        agente.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            return total_profit, history


def show_eval_result(model_name, profit, initial_offset):
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{} previu lucro de: {}\n'.format(model_name, format_position(profit)))


def switch_k_backend_device():
    """ Altera o `keras` backend de GPU para CPU se necessário.
    (Com GPU é necessário instalar tensorflow-gpu e plataforma CUDA). 
    """
    if K.backend() == 'tensorflow':
        logging.debug('Mudando para TensorFlow com CPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    coloredlogs.install(level='DEBUG')
    try:
        main()
    except KeyboardInterrupt:
        print('Aborted')
