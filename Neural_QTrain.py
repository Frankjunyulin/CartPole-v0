import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100 # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.9# discount factor
INITIAL_EPSILON =  0.45# starting value of epsilon
FINAL_EPSILON =  0.01# final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
batch_size = 128
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
replay_buffer = []

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
layer_one_out = tf.layers.dense(state_in, 64, activation = tf.nn.relu)
layer_two_out = tf.layers.dense(layer_one_out, 64,activation = tf.nn.relu)
layer_three_out = tf.layers.dense(layer_two_out, 64,activation = tf.nn.relu)
"""
w1 = tf.Variable(tf.random_normal([STATE_DIM, 20]))
b1 = tf.Variable(tf.zeros([20]) + 0.1)
l1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)
w2 = tf.Variable(tf.random_normal([20, ACTION_DIM]))
b2 = tf.Variable(tf.zeros([ACTION_DIM]) + 0.1)
q_values = tf.matmul(l1, w2) + b2
"""
#conv1 = tf.layers.conv2d(state_in, 128, activation = tf.nn.relu)
#conv2 = tf.layers.conv2d(conv1, 128, activation = tf.nn.relu)


# TODO: Network outputs
q_values =tf.layers.dense(layer_three_out, ACTION_DIM)
#print(q_values)
q_selected_action = tf.reduce_sum(tf.multiply(q_values, action_in),reduction_indices=1)
# TODO: Loss/Optimizer Definition
loss =tf.reduce_mean(tf.square(target_in - q_selected_action))
#0.00022
#0.00018
optimizer =tf.train.AdamOptimizer(0.000180).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()
    # Update epsilon once per episode
    epsilon -= epsilon/ EPSILON_DECAY_STEPS
#    epsilon = 0
    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))


   #     nextstate_q_values = q_values.eval(feed_dict={
    #        state_in: [next_state]
     #   })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated

   #     one_hot_action = np.zeros(ACTION_DIM);
    #    one_hot_action[np.argmax(action)] = 1
        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > 10000:
          #  print("update")
            replay_buffer.pop(0)
         #   replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >=batch_size:
            minibatch = random.sample(replay_buffer,batch_size)

            states = [data[0] for data in minibatch]
            actions = [data[1] for data in minibatch]
            rewards = [data[2] for data in minibatch]

            next_state_batch = [data[3] for data in minibatch]
            nextstate_q_values = q_values.eval(feed_dict={state_in: next_state_batch})
            target = []
            for i in range(0, batch_size):
                terminated = minibatch[i][4]
                if terminated:
                    #pass
                    target.append(-1.0)
                else:
                    target_val = rewards[i] + GAMMA * np.max(nextstate_q_values[i])
                    target.append(target_val)

            session.run([optimizer], feed_dict={
                target_in: target,
                action_in: actions,
                state_in: states
            })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
