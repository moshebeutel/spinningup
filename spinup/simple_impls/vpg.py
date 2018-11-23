import tensorflow as tf
import numpy as np
import gym

# Global variables
env_name = 'CartPole-v0'
num_of_train_iterations = 50
hidden_dim = 32
number_of_layers=2
learning_rate = 0.0001
batch_size=5000
discount_factor = 0.99


#init
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
env.reset()
done = False
step = 0

# model - build computation graph
with tf.variable_scope('model'):
    #input layer
    obs_ph = tf.placeholder(dtype=np.float32, shape=(None,obs_dim),name='obs_ph')
    #mlp (Multi Layer Perceptron) - hidden layers
    #define sizes for hidden layers
    hidden_sizes = [hidden_dim] * number_of_layers
    #build hidden layers
    for h in hidden_sizes:
        x = tf.layers.dense(obs_ph, units=h,activation=tf.tanh)
    #logits
    logits = tf.layers.dense(x, units = env.action_space.n, activation=None)
    #actions - output layer
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

#define loss function

rtgs_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='rtgs_ph')
acts_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='acts_ph')
action_mask = tf.one_hot(acts_ph, env.action_space.n)
log_probs = tf.reduce_sum(action_mask * tf.nn.softmax(logits=logits), axis=1)
loss = -tf.reduce_mean(rtgs_ph * log_probs) 

#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#session init
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

#train
for iter in range(num_of_train_iterations):
    #single train iteration

    #reset batch data
    batch_obs, batch_acts, batch_rtgs, batch_rets, batch_lens = [], [], [], [], []
    #reset episodic data
    obs, rew, done, ep_rews = env.reset(), 0, False, []
    while len(batch_obs) < batch_size:
        #record batch
        env.render()
        batch_obs.append(obs.copy())
        selected_action  = session.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
        obs, reward, done, _ = env.step(selected_action)
        batch_acts.append(selected_action)
        ep_rews.append(reward)
        if done:
            #end of episode
            batch_rets.append(sum(ep_rews))
            batch_lens.append(len(ep_rews))
            #calc reward to go
            n = len(ep_rews)
            x = np.array(ep_rews)
            #prepare array of powers of dicount factor
            y = discount_factor ** np.arange(n) 
            rtgs_array = np.zeros_like(x, dtype=np.float32)
            for step in range(n):
                rtgs_array[step] = sum(x[step:]*y[:n-step])
            batch_rtgs += list(rtgs_array)
            #reset episodic data
            obs, rew, done, ep_rews = env.reset(), 0, False, []
            

    # TODO: normalize advs trick:
    #batch_advs = np.array(batch_rtgs)
    #batch_advs = (batch_advs - np.mean(batch_advs))/(np.std(batch_advs) + 1e-8)
    feed_dict={obs_ph: np.array(batch_obs[:len(batch_rtgs)]), acts_ph: np.array(batch_acts[:len(batch_rtgs)]), rtgs_ph: np.array(batch_rtgs)}
    batch_loss, _ = session.run([loss, optimizer], feed_dict=feed_dict)

    print('iteration number: %d \t loss: %.3f \t return: %.3f \t epispode mean length: %.3f'%
            (iter, batch_loss, np.mean(batch_rets), np.mean(batch_lens))) 


# #play
# while not done:
#     step += 1
#     env.render()
#     observation, reward, done, info =  env.step(env.action_space.sample())
#     print(step, observation, reward, done, info)

# print('Game Over afetr {} steps'.format(step))


