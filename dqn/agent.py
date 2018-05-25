from image_processor import ImageProcessor
from collections import deque
import numpy as np
import random
from dqn.memory import Memory
from dqn.model import DQNetwork
import tensorflow as tf


class Agent:
    def __init__(self, game, env, writer, viewer=None):
        self.game = game
        self.model = DQNetwork(game.frame_shape, game.action_size)
        self.env = env
        self.memory = Memory(max_size=self.model.memory_size)
        self.writer = writer
        self.decay_step = 0

        tf.summary.scalar("Loss", self.model.loss)
        self.write_op = tf.summary.merge_all()

        training_config = game.network_configs.get(self.model.name)
        self.batch_size = training_config.get('batch_size')
        self.max_steps = training_config.get('max_steps')
        self.total_episodes = training_config.get('total_episodes')

        self.processor = ImageProcessor(game.process_image_format, render_format=game.render_format, viewer=viewer)
        self.stacked_frames = deque([np.zeros(game.frame_shape, dtype=np.int) for i in range(self.model.stack_size)], maxlen=4)

    def prepare(self):
        self.env.reset()
        for i in range(self.batch_size):
            if self.env.img is not None:
                state = ImageProcessor.stack_frames(self.stacked_frames, self.processor.process_image(self.env.img))
            else:
                state = np.zeros(self.game.stacked_frame_shape, dtype=np.int)

            ac = random.choice(self.game.action_space)
            ob, rew, done, info = self.env.step(ac)

            if done:
                # We finished the episode
                next_state = np.zeros(self.game.stacked_frame_shape, dtype=np.int)

                # Add experience to memory
                self.memory.add((state, self.game.convert_action(low=ac), rew, next_state, done))

                # Start a new episode
                self.env.reset()
            else:
                # Get the next state
                next_state = self.processor.process_image(ob)
                next_state = self.processor.stack_frames(self.stacked_frames, next_state)

                # Add experience to memory
                self.memory.add((state, self.game.convert_action(low=ac), rew, next_state, done))

    def train(self, sess, episode, quiet=False):
        # Make new episode
        self.env.reset()
        total_reward = 0
        step = 0

        # Observe the first state
        frame = self.processor.process_image(self.env.img)
        state = self.processor.stack_frames(self.stacked_frames, frame)

        while step < self.max_steps:
            step += 1
            # Increase decay_step
            self.decay_step += 1

            # EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            # First we randomize a number
            exp_exp_tradeoff = np.random.rand()

            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = self.model.explore_stop + (self.model.explore_start - self.model.explore_stop) * \
                np.exp(-self.model.decay_rate * self.decay_step)

            if explore_probability > exp_exp_tradeoff:
                # Make a random action
                action = random.choice(self.game.possible_actions)
            else:
                # Get action from Q-network
                # Estimate the Qs values state
                Qs = sess.run(self.model.output, feed_dict={self.model.inputs_: state.reshape((1, *state.shape))})

                # Take the biggest Q value (= the best action)
                action = np.argmax(Qs)
                action = self.game.possible_actions[int(action)]
            # Do the action
            prev_state = self.env.img
            next_state, reward, done, info = self.env.step(self.game.convert_action(high=action))
            while np.array_equal(self.processor.process_image(prev_state), self.processor.process_image(next_state)):
                step += 1
                next_state, reward, done, info = self.env.step(self.game.convert_action(high=action))
            total_reward += reward
            if quiet:
                print('Ep {}, step {}'.format(episode, step), info)
            else:
                self.processor.render(self.env.img)

            # If the game is finished
            if done or info.get('lap') == 1:
                # the episode ends so no next state
                next_state = np.zeros(self.game.frame_shape, dtype=np.int)
                next_state = self.processor.stack_frames(self.stacked_frames, next_state)

                # Set step = max_steps to end the episode
                step = self.max_steps

                self.memory.add((state, action, reward, next_state, done))

            else:
                # Get the next state
                next_state = self.processor.stack_frames(self.stacked_frames, self.processor.process_image(next_state))

                # Add experience to memory
                self.memory.add((state, action, reward, next_state, done))

            # LEARNING PART
            # Obtain random mini-batch from memory
            batch = self.memory.sample(self.batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            dones = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            target_Qs = sess.run(self.model.output, feed_dict={self.model.inputs_: next_states})

            # Set Qhat = r if the episode ends at +1, otherwise set Qhat = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards[i])
                else:
                    target = rewards[i] + self.model.gamma * np.max(target_Qs[i])
                    target_Qs_batch.append(target)

            targets = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([self.model.loss, self.model.optimizer],
                               feed_dict={self.model.inputs_: states,
                                          self.model.target_Q: targets,
                                          self.model.actions_: actions})

            # Write TF Summaries
            summary = sess.run(self.write_op, feed_dict={self.model.inputs_: states,
                                                         self.model.target_Q: targets,
                                                         self.model.actions_: actions})
            self.writer.add_summary(summary, episode)
            self.writer.flush()

        self.print_progress(episode, total_reward, loss, explore_probability)

        return sess

    def play(self, sess):
        self.env.reset()
        done = False
        while not done:
            self.processor.render(self.env.img)
            frame = self.processor.process_image(self.env.img)
            state = self.processor.stack_frames(self.stacked_frames, frame)
            # Take the biggest Q value (= the best action)
            Qs = sess.run(self.model.output, feed_dict={self.model.inputs_: state.reshape((1, *state.shape))})
            action = np.argmax(Qs)
            action = self.game.action_space[int(action)]
            next_state, reward, done, info = self.env.step(action)
            while np.array_equal(self.game.frame_shape, self.processor.process_image(next_state)):
                next_state, reward, done, info = self.env.step(action)
        print(info)

    @staticmethod
    def print_progress(episode, total_reward, loss, explore_probability):
        print('Episode: {}'.format(episode),
              'Total reward: {}'.format(total_reward),
              'Training loss: {:.4f}'.format(loss),
              'Explore P: {:.4f}'.format(explore_probability))
