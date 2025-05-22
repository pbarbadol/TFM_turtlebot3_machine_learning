#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee
# Modified for DDQN by Pablo Barbado Lozano

import collections
import datetime
import json
import math
import os
import random
import sys
import time

from keras.api.layers import Dense
from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.optimizers import Adam
import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow

from turtlebot3_msgs.srv import Dqn

SEED = 42

random.seed(SEED)
numpy.random.seed(SEED)

tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%Y%m%d-%H%M%S]') # Year added for better sorting


class DQNMetric(tensorflow.keras.metrics.Metric): # Can keep this name

    def __init__(self, name='dqn_metric'): # Can keep this name
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)


class DDQNAgent(Node): # Renamed class

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('ddqn_agent') # Renamed node for clarity

        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 24 + 4
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 20000
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_memory = collections.deque(maxlen=500000)
        self.min_replay_memory_size = 5000

        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000 # Steps in environment (or training calls, check logic)
        self.target_update_after_counter = 0

        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        # --- MODIFIED for DDQN model naming ---
        self.model_base_name = 'ddqn_stage' + str(self.stage) + '_episode'
        self.model_path = os.path.join(
            self.model_dir_path,
            self.model_base_name + str(self.load_episode) + '.h5'
        )

        if self.load_model:
            self.model.set_weights(load_model(self.model_path).get_weights())
            with open(os.path.join(
                self.model_dir_path,
                self.model_base_name + str(self.load_episode) + '.json'
            )) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                self.step_counter = param.get('step_counter')
        # --- END MODIFICATION ---

        if LOGGING:
            # --- MODIFIED for DDQN log naming ---
            tensorboard_file_name = current_time + ' ddqn_stage' + str(self.stage) + '_reward'
            # --- END MODIFICATION ---
            dqn_reward_log_dir = 'logs/gradient_tape/' + tensorboard_file_name
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step += 1

                # Q-values for action selection are still from the online model
                q_values_for_action_selection = self.model.predict(state)
                sum_max_q += float(numpy.max(q_values_for_action_selection))

                action = int(self.get_action(state)) # get_action uses self.model
                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done) # DDQN logic is inside train_model

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode_num
                            ) # Metric name 'dqn_reward' is fine for TensorBoard tag
                        self.dqn_reward_metric.reset_states()

                    print(
                        f'DDQN - Episode: {episode}, '
                        f'score: {score:.2f}, '
                        f'memory: {len(self.replay_memory)}, '
                        f'epsilon: {self.epsilon:.4f}, '
                        f'steps: {self.step_counter}'
                    )

                    param_keys = ['epsilon', 'step_counter'] # Corrected key name
                    param_values = [self.epsilon, self.step_counter]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                time.sleep(0.01)

            if self.train_mode:
                if episode % 100 == 0:
                    # --- MODIFIED for DDQN model saving ---
                    self.model_path = os.path.join(
                        self.model_dir_path,
                        self.model_base_name + str(episode) + '.h5')
                    self.model.save(self.model_path)
                    with open(
                        os.path.join(
                            self.model_dir_path,
                            self.model_base_name + str(episode) + '.json'
                        ),
                        'w'
                    ) as outfile:
                        json.dump(param_dictionary, outfile)
                    # --- END MODIFICATION ---

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                f'Exception while calling service: {future.exception()}')
            # Handle error, maybe by retrying or exiting
            state = numpy.zeros((1, self.state_size)) # Placeholder
        return state

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1 # Increment step counter here, as it's per environment step
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay)
            
            # Epsilon-greedy action selection
            if random.random() > self.epsilon: # Exploit (Original had lucky > (1-epsilon) for random, fixed to common way)
                q_values = self.model.predict(state) # Use online model for action selection
                result = numpy.argmax(q_values[0])
            else: # Explore
                result = random.randint(0, self.action_size - 1)
        else: # Test mode
            q_values = self.model.predict(state)
            result = numpy.argmax(q_values[0])
        return result

    def step(self, action):
        req = Dqn.Request()
        req.action = action # action is already an int
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                f'Exception while calling service: {future.exception()}')
            # Handle error appropriately, e.g., return a default state, reward, done
            next_state = numpy.zeros((1, self.state_size)) # Placeholder
            reward = 0.0
            done = True # Assume failure if service call fails
        return next_state, reward, done

    def create_qnetwork(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # Linear activation for Q-values
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary(print_fn=self.get_logger().info) # Log model summary via ROS logger
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.get_logger().info('*Target model updated*')
        # self.target_update_after_counter = 0 # Reset counter if update is based on this

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal): # terminal means 'done'
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)

        current_states_list = [transition[0].squeeze() for transition in data_in_mini_batch]
        actions_list = [transition[1] for transition in data_in_mini_batch]
        rewards_list = [transition[2] for transition in data_in_mini_batch]
        next_states_list = [transition[3].squeeze() for transition in data_in_mini_batch]
        done_list = [transition[4] for transition in data_in_mini_batch]

        current_states_batch = numpy.array(current_states_list)
        next_states_batch = numpy.array(next_states_list)

        # Predict Q values for current states using the online model
        current_q_values_batch = self.model.predict(current_states_batch)

        # --- DDQN IMPLEMENTATION ---
        # 1. Select best actions for next_states using the ONLINE model
        next_actions_online_model = self.model.predict(next_states_batch)
        argmax_next_actions = numpy.argmax(next_actions_online_model, axis=1)

        # 2. Evaluate these selected actions using the TARGET model
        next_q_values_target_model_batch = self.target_model.predict(next_states_batch)
        
        target_q_values = numpy.copy(current_q_values_batch)

        for i in range(self.batch_size):
            if done_list[i]:
                target_q_values[i, actions_list[i]] = rewards_list[i]
            else:
                # Q_target = R + gamma * Q_target_network(S', argmax_a'(Q_online_network(S', a')))
                selected_q_value_by_target = next_q_values_target_model_batch[i, argmax_next_actions[i]]
                target_q_values[i, actions_list[i]] = rewards_list[i] + self.discount_factor * selected_q_value_by_target
        # --- END DDQN IMPLEMENTATION ---

        self.model.fit(
            current_states_batch,
            target_q_values,
            batch_size=self.batch_size,
            verbose=0,
            epochs=1 # Typically 1 epoch for RL batch update
        )
        
        # Logic for updating target model based on environment steps (step_counter)
        # This is often preferred over updating based on number of train_model calls
        if self.step_counter % self.update_target_after == 0:
             self.update_target_model()


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    rclpy.init(args=args)

    # Instantiate the DDQNAgent
    ddqn_agent_node = DDQNAgent(stage_num, max_training_episodes) # Use the new class name
    
    try:
        rclpy.spin(ddqn_agent_node)
    except KeyboardInterrupt:
        ddqn_agent_node.get_logger().info("Shutting down DDQN agent node.")
    finally:
        ddqn_agent_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()