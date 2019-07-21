# Módulo com a definição do Agente Inteligente

# Imports
import random
import numpy as np
import tensorflow as tf
import keras.backend as K

from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop
from keras.initializers import VarianceScaling

def huber_loss(y_true, y_pred, clip_delta = 1.0):
    """ Huber loss - Custom Loss Function para Q Learning """
    """ https://en.wikipedia.org/wiki/Huber_loss """ 
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

class Agente:

    def __init__(self, state_size, pretrained = False, model_name = None):

        ''' Parâmetros do Agente '''
        self.state_size = state_size    	
        self.action_size = 3           		# [espera, compra, vende]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=1000)
        self.first_iter = True

        ''' Parâmetros do Modelo '''
        self.model_name = model_name
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {'huber_loss': huber_loss}  
        self.optimizer = RMSprop(lr = self.learning_rate)
        self.initializer = VarianceScaling()

        ''' Carrega o modelo pré-treinado '''
        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

    def _model(self):
        """	Cria o modelo """
        model = Sequential()
        model.add(Dense(units=24, input_dim = self.state_size, kernel_initializer = self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=64, kernel_initializer = self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=64, kernel_initializer = self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units=24, kernel_initializer = self.initializer))
        model.add(Activation('relu'))
        model.add(Dense(units = self.action_size, kernel_initializer = self.initializer))

        model.compile(loss = self.loss, optimizer = self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """ Adiciona dados relevantes na memória do agente """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval = False):
        """ Toma ações a partir de um possível conjunto de ações """
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1

        # Grava as previsões de ações do agente
        options = self.model.predict(state)

        # argmax retorna a posição do maior valor
        return np.argmax(options[0])

    def train_experience_replay(self, batch_size):
        """ Treina com as experiências anteriores na memória. """
        mini_batch = random.sample(self.memory, batch_size)

        # Cria os datasets de treino vazios
        X_train, y_train = [], []

        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)

            # Array com cada ação do agente e cada recompensa recebida
            target_f[0][action] = target

            # Preenche dados de entrada (x) e de saída (y)
            X_train.append(state[0])
            y_train.append(target_f[0])

        history = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, epoch):
        self.model.save('models/{}_{}'.format(self.model_name, epoch))

    def load(self):
        return load_model('models/' + self.model_name, custom_objects = self.custom_objects)