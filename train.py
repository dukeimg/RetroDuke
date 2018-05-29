#!/usr/bin/env python

import datetime
import subprocess

import argparse
import retro
import tensorflow as tf

from game import Game

parser = argparse.ArgumentParser()
parser.add_argument('game', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('network', nargs='?', help='the algorithm to work with (dqn, pqn)')
parser.add_argument('--state', help='the initial state file to load, minus the extension')
parser.add_argument('--model', '-m', help='the model to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--quiet', '-q', default=False, help='do not render gameplay')
args = parser.parse_args()

viewer = None
if not args.quiet:
    from gym.envs.classic_control.rendering import SimpleImageViewer
    viewer = SimpleImageViewer()

game = Game(args.game)
env = retro.make(args.game, args.state or retro.STATE_DEFAULT, scenario=args.scenario)

# Reset the graph
tf.reset_default_graph()

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/{}".format(args.network))

if args.network == 'dqn':
    from dqn.agent import Agent

    agent = Agent(game, env, writer, viewer=viewer)
    agent.prepare()
elif args.network == 'pgn':
    from pgn.agent import Agent

    agent = Agent(game, env, writer, viewer=viewer)
else:
    raise ValueError('Please specify a valid network (dqn, pqn)')

# Saver will help us to save our model
saver = tf.train.Saver()

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 0
config.inter_op_parallelism_threads = 0
with tf.Session(config=config) as sess:
    # Initialize the variables
    if args.model:
        saver.restore(sess, "./models/{}/{}.ckpt".format(args.network, args.model))
    else:
        sess.run(tf.global_variables_initializer())

    for episode in range(agent.total_episodes):
        sess, save = agent.train(sess, episode, quiet=args.quiet)

        if episode % 5 == 0:
            filename = datetime.datetime.now().strftime('ep-{}-%Y-%m-%d %H:%M:%S'.format(episode))
            saver.save(sess, "./models/{}/{}.ckpt".format(agent.model.name, filename))
        if save:
            saver.save(sess, "./models/{}/{}.ckpt".format(agent.model.name, save))
        saver.save(sess, "./models/latest_{}_model.ckpt".format(agent.model.name))
        print("Model Saved")
        if not args.quiet:
            subprocess.run(['python', 'play.py', args.game, args.network])
