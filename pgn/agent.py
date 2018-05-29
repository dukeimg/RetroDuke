from image_processor import ImageProcessor
from collections import deque
import numpy as np
from pgn.model import PGNetwork
import tensorflow as tf


class Agent:
    def __init__(self, game, env, writer, viewer=None):
        self.game = game
        self.model = PGNetwork(game.frame_shape, game.action_size)
        self.env = env
        self.writer = writer
        self.all_rewards = []

        tf.summary.scalar("Loss", self.model.loss)
        self.write_op = tf.summary.merge_all()

        training_config = game.network_configs.get(self.model.name)
        self.batch_size = training_config.get('batch_size')
        self.max_steps = training_config.get('max_steps')
        self.total_episodes = training_config.get('total_episodes')
        self.learning_rate_decay = (self.model.alpha_start - self.model.alpha_stop) / self.total_episodes

        self.processor = ImageProcessor(game.process_image_format, render_format=game.render_format, viewer=viewer)
        self.stacked_frames = deque([np.zeros(game.frame_shape, dtype=np.int) for _ in range(self.model.stack_size)],
                                    maxlen=4)

    def discount_and_normalize_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.model.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards) or 1
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

    def make_batch(self, sess):
        # Initialize lists: states, actions, rewards, discountedRewards
        states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []

        # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per
        # episode)
        episode_num = 1

        # Launch a new episode
        self.env.reset()

        # Get a new state
        frame = self.processor.process_image(self.env.img)
        state = self.processor.stack_frames(self.stacked_frames, frame)

        step = 0
        while True:
            step += 1
            # Run State Through Policy & Calculate Action
            action_probability_distribution = sess.run(
                self.model.action_distribution,
                feed_dict={
                    self.model.inputs_: state.reshape(1, *self.game.frame_shape, self.model.stack_size)
                })
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = self.game.possible_actions[action]

            # Perform action
            prev_state = self.env.img
            next_state, reward, done, info = self.env.step(self.game.convert_action(high=action))
            while np.array_equal(self.processor.process_image(prev_state), self.processor.process_image(next_state)):
                next_state, reward, done, info = self.env.step(self.game.convert_action(high=action))

            self.processor.render(self.env.img)

            # Store results
            states.append(state)
            rewards.append(reward)

            # For actions because we output only one (the index) we need (None, 3) (1 is for the action taken)
            # action_ = np.zeros((action_size, action_size))
            # action_[action][action] = 1

            actions.append(action)

            if done or step > self.max_steps:
                step = 0
                # the episode ends so no next state
                rewardsFeed.append(rewards)

                # Calculate gamma Gt
                discountedRewards.append(self.discount_and_normalize_rewards(rewards))

                if len(np.concatenate(rewardsFeed)) > self.batch_size:
                    break

                # Reset the transition stores
                rewards = []

                # Add episode
                episode_num += 1

                # New episode
                self.env.reset()

            # If not done, the new_state become the current state
            new_state = self.processor.process_image(self.env.img)
            state = self.processor.stack_frames(self.stacked_frames, new_state)

        return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewardsFeed), np.concatenate(
            discountedRewards), episode_num

    def train(self, sess, episode, quiet=False):
        mean_reward_total = []
        # Gather training data

        batch_states, batch_actions, batch_rewards, batch_discounted_rewards, batch_number_of_episodes = \
            self.make_batch(sess)

        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(batch_rewards)
        self.all_rewards.append(total_reward_of_that_batch)

        # Calculate the mean reward of the batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, batch_number_of_episodes)
        mean_reward_total.append(mean_reward_of_that_batch)
        mean_reward = np.divide(np.sum(mean_reward_total), episode)

        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), episode)

        # Calculate maximum reward recorded
        maximum_reward_recorded = np.amax(self.all_rewards)

        # Feedforward, gradient and backpropagation
        loss_, _ = sess.run(
            [self.model.loss, self.model.train_opt],
            feed_dict={
                self.model.inputs_: batch_states.reshape((len(batch_states), *self.game.frame_shape, self.model.stack_size)),
                self.model.actions: batch_actions,
                self.model.discounted_episode_rewards_: batch_discounted_rewards
            }
        )

        # Write TF Summaries
        summary = sess.run(
            self.write_op,
            feed_dict={
                self.model.inputs_: batch_states.reshape((len(batch_states),  *self.game.frame_shape, self.model.stack_size)),
                self.model.actions: batch_actions,
                self.model.discounted_episode_rewards_: batch_discounted_rewards,
                self.model.mean_reward_: mean_reward
            }
        )

        self.writer.add_summary(summary, episode)
        self.writer.flush()

        self.print_progress(episode, batch_number_of_episodes, total_reward_of_that_batch, mean_reward_of_that_batch,
                            average_reward_of_all_training, maximum_reward_recorded)

        self.model.learning_rate -= self.learning_rate_decay

        print(self.model.learning_rate)

        if total_reward_of_that_batch == maximum_reward_recorded:
            return sess, 'max_reward'
        else:
            return sess, None

    def play(self, sess):
        self.env.reset()
        done = False
        while not done:
            frame = self.processor.process_image(self.env.img)
            state = self.processor.stack_frames(self.stacked_frames, frame)

            # Run State Through Policy & Calculate Action
            action_probability_distribution = sess.run(
                self.model.action_distribution, feed_dict={
                    self.model.inputs_: state.reshape(1, *self.game.frame_shape, self.model.stack_size)
                }
            )
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = self.game.action_space[action]
            next_state, reward, done, info = self.env.step(action)
            while np.array_equal(self.game.frame_shape, self.processor.process_image(next_state)):
                next_state, reward, done, info = self.env.step(action)
            self.processor.render(self.env.img)
        print(info)

    def print_progress(self, epoch, batch_number_of_episodes, total_reward_of_that_batch, mean_reward_of_that_batch,
                       average_reward_of_all_training, maximum_reward_recorded):
        print("==========================================")
        print("Epoch: ", epoch, "/", self.total_episodes)
        print("-----------")
        print("Number of training episodes: {}".format(batch_number_of_episodes))
        print("Total reward: {}".format(total_reward_of_that_batch))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximum_reward_recorded))
