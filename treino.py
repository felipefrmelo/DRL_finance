# Módulo com o treinamento do Agente

# Imports
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
from executa import evaluate_model
from utils import (get_state, get_stock_data, format_currency, format_position)

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


@click.command()
@click.option(
    '-ts',
    '--treina-stock',
    type=click.Path(exists=True),
    default='data/GOOGL.csv',
    help='Dados de treinamento'
)
@click.option(
    '-vs',
    '--val-stock',
    type=click.Path(exists=True),
    default='data/GOOGL_2018.csv',
    help='Dados de validação'
)
@click.option(
    '-ws',
    '--window-size',
    default=10,
    help='Tamanho da janela n-dias dos estados anteriores para normalização'
)
@click.option(
    '-bz',
    '--batch-size',
    default=16,
    help='Tamanho do mini-batch size para usar no treinamento do agente'
)
@click.option(
    '-ep',
    '--ep-count',
    default=50,
    help='Número de épocas para treinar o agente'
)
@click.option(
    '-mn',
    '--model-name',
    default='modelo_stock',
    help='Nome do modelo para salvar ou fazer checkpoint'
)
@click.option(
    '-pre',
    '--pretrained',
    is_flag=True,
    help='Se o modelo é pré-treinado'
)
@click.option(
    '-d',
    '--debug',
    is_flag=True,
    help='Flag para o modo debug (imprime posição em cada passo durante avaliação)'
)
def main(treina_stock, val_stock, window_size, batch_size, ep_count, model_name, pretrained, debug):
    """ Treina o bot de previsão de compra e venda de ações usando Deep Q-Learning. """

    # Define o dispositivo
    switch_k_backend_device()

    # Cria instância do agente
    agente = Agente(window_size, pretrained = pretrained, model_name = model_name)

    # Dados de treino do modelo
    train_data = get_stock_data(treina_stock)

    # Dados de validação
    val_data = get_stock_data(val_stock)

    # Validação inicial
    initial_offset = val_data[1] - val_data[0]

    # Para cada época
    for epoch in range(1, ep_count + 1):
        train_result = train_model(agente, epoch, train_data, ep_count = ep_count, batch_size = batch_size, window_size = window_size)
        val_result, _ = evaluate_model(agente, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


def train_model(agente, epoch, data, ep_count = 100, batch_size = 32, window_size = 10):
    data_length = len(data) - 1

    # Lucro de compra e venda
    total_profit = 0

    agente.inventory = []
    avg_loss = []
    start = clock()
    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total = data_length, leave = True, desc = 'Epoch {}/{}'.format(epoch, ep_count)):
        action = agente.act(state)
        
        # Espera
        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0
        
        # Compra
        if action == 1:
            agente.inventory.append(data[t])
        
        # Vende
        elif action == 2 and len(agente.inventory) > 0:
            bought_price = agente.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price

        done = True if t == data_length - 1 else False
        agente.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agente.memory) > batch_size:
            loss = agente.train_experience_replay(batch_size)
            avg_loss.append(loss)
        if done:
            end = clock() - start

    # Checkpoint
    if epoch % 10 == 0:
        agente.save(epoch)
    return (epoch, ep_count, total_profit, np.mean(np.array(avg_loss)), end)


def show_train_result(result, val_position, initial_offset):
    """ Mostra os resultados de treinamento. """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Epoch {}/{} - Posição em Treino: {}  Posição em Validação: USELESS  Perda em Treino: {:.4f}  (~{:.4f} segundos)'
                     .format(result[0], result[1], format_position(result[2]), result[3], result[4]))
    else:
        logging.info('Epoch {}/{} - Posição em Treino: {}  Posição em Validação: {}  Perda em Treino: {:.4f}  (~{:.4f} segundos)'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3], result[4]))


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
        print('Aborted!')
