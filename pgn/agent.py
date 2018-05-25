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