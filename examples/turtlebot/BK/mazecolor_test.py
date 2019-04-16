#!/usr/bin/env python

'''
Based on: 
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''

import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import memory
import cv2

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        predicted = self.targetModel.predict(state.reshape(1,len(state)))

        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples        
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print file
        os.unlink(file)

def writeFile(text, path):
    with open(path, "w") as myfile:
        myfile.write(str(text) + "\n")

def processImage(image):
    red = image.copy()
    red[:,:,1] = 0
    red[:,:,2] = 0
    gray = cv2.cvtColor(red, cv2.COLOR_RGB2GRAY)
    # maxVal = np.amax(np.asarray(gray))
    maxVal = 76 # from experience 
    ret, thresh = cv2.threshold(gray, maxVal-1, maxVal, cv2.THRESH_BINARY)
    return thresh, maxVal

def getPoints(image, red, green, blue):    
    height, width, depth = image.shape
    sumX = 0
    sumY = 0
    countX = 0
    countY = 0 
    mid = float(width/2)
    for x in range(height):
        for y in range(width):
            if (image[x][y][0] == red) and (image[x][y][1] <= green) and (image[x][y][2] <= blue):
                sumX += x
                weightY = abs(float(y - mid)) / mid
                sumY += y * weightY
                countX += 1
                countY += weightY

    if countX == 0 and countY == 0:
        return np.asarray([0, 0])

    centerX = float(sumX)/countX
    centerY = float(sumY)/countY

    return np.asarray([centerX, centerY])

def getTargetPoints(image):
    return getPoints(image, 102, 20, 20)

def getHintPoints(image):
    return getPoints(image, 255, 120, 120)

def getImageAction(height, width, x, y):
    if x == 0 and y == 0:
        return 100
    mid = width/2
    return np.rint((y - mid) * 10 / mid + 10)

def checkDanger(laser):
    for distance in laser:
        if distance < 0.5:
            return True
    return False


def normalize(array):
    epsilon = 1e-4
    max = np.amax(array)
    min = np.amin(array)
    return (array - min + epsilon)/(max - min + epsilon)


if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboTurtlebotMazeColor-v0')
    # outdir = '/tmp/gazebo_gym_experiments/'


    weights_path = '/home/tuanguyen/Desktop/model/turtle_mazecolor_camera_dqn.h5'
    params_json  = '/home/tuanguyen/Desktop/model/turtle_mazecolor_camera_dqn.json'


    #Load weights, monitor info and parameter info.
    #ADD TRY CATCH for this else
    with open(params_json) as outfile:
        d = json.load(outfile)
        epochs = d.get('epochs')
        steps = d.get('steps')
        updateTargetNetwork = d.get('updateTargetNetwork')
        explorationRate = d.get('explorationRate')
        minibatch_size = d.get('minibatch_size')
        learnStart = d.get('learnStart')
        learningRate = d.get('learningRate')
        discountFactor = d.get('discountFactor')
        memorySize = d.get('memorySize')
        network_inputs = d.get('network_inputs')
        network_outputs = d.get('network_outputs')
        network_layers = d.get('network_structure')
        current_epoch = d.get('current_epoch')

    deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    deepQ.initNetworks(network_layers)
    deepQ.loadWeights(weights_path)

    observation = env.reset()

    rotateDirection = 21
    pointPassed = []
    lastAction = 0
    totalRotation = 0


    while True:
        [image, laser, position] = observation

        # get image info
        x, y = getTargetPoints(image)
        if x == 0 and y == 0:
            x, y = getHintPoints(image)

        height, width, depth = image.shape
        imageAction = getImageAction(height, width, x, y)
        normalizedLaser = normalize(laser)
        totalState = np.append(normalizedLaser, np.asarray([x/height, y/width]))

        qValues = deepQ.getQValues(totalState) # get Q-values
        laserAction = deepQ.selectAction(qValues, 0)
        action = laserAction


        # print([imageAction, laserAction])
        # if imageAction == 100:
        #     # action = laserAction
        #     prob = np.random.rand()
        #     if (prob < 0.5):
        #         action = 21
        #     else:
        #         action = -1

        #     if lastAction == 21 or lastAction == -1:
        #         action = lastAction
        #         totalRotation += 0.2

        # else:
        #      # if the last action is rotating
        #     if lastAction == 21 or lastAction == -1:
        #         # check if the current hint/target is already seen
        #         if totalRotation > 2.5:
        #             # rotate at opposing direction
        #             action = 20 - lastAction
        #             totalRotation = 0
        #         else:
        #             # follow new hint/target
        #             action = 20 - imageAction 
        #             totalRotation = 0
        #     else:
        #         action = 20 - imageAction
        #         totalRotation = 0

        # if imageAction == 100:
        #     action = laserAction
        # else:
        #     action = int((20 - imageAction) * 2/3 + laserAction * 1/3)


        # if imageAction == 100:
        #     action = laserAction
        # else:
        #     if imageAction >= 9 and imageAction <= 11:
        #         action = 20 - imageAction
        #     else:
        #         action = laserAction
        # isDanger = checkDanger(laser)
        # if isDanger:
        #     print('In danger')
        #     action = laserAction
        # else:
        #     print([imageAction, laserAction])
        #     if imageAction == 100:
        #         action = laserAction
        #     else:
        #         action = int((20 - imageAction) * 1/2 + laserAction * 1/2)
        #         # action = 20 - imageAction

        observation, reward, done, info = env.step(action)
        lastAction = action
        if done:

            observation = env.reset()

    env.close()
