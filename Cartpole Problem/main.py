from os import system
import gym
import tensorflow as tf
import tflearn
import random
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 700
score_requirement = 100
initial_games = 100000

def random_game():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break

def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation = observation

			score += reward

			if done:
				break

		if score >= score_requirement:
			accepted_scores.append(score)

			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	print ('Averaged accepted scores: {}'.format(mean(accepted_scores)))
	print ('Median accepted scores: {}'.format(median(accepted_scores)))
	print (Counter(accepted_scores))

	return training_data

def ANN_model(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 64, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 64, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]

	if not model:
		model = ANN_model(len(X[0]))

	model.fit({'input': X}, {'targets': y}, n_epoch=5, show_metric=True, snapshot_step=500, run_id='Cartpole')
	return model

print ('Beginning Training Session')
training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

games = 20

for each_game in range(games):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()

	print ('Starting game {}'.format(each_game+1))

	for _ in range(goal_steps):
		env.render()

		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation, action])
		score += reward
		if done:
			break

	print ('Score: {}'.format(score))
	scores.append(score)

print ('Average score: {}'.format(sum(scores)/len(scores)))
print ('Maximun score: {}'.format(max(scores)))
print ('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
