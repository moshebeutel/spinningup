import tensorflow as tf
import numpy as np
import gym
from spinup.utils.logx import EpochLogger
# from pdb import set_trace as bp


def simple_vpg(env_fn=lambda : gym.make('CartPole-v1')
    , actor_critic=None
    , ac_kwargs=dict()
    , seed=0
    , steps_per_epoch=5000
    , epochs=10000
    , gamma=0.99
    , pi_lr=1e-5
    , logger_kwargs=dict()
    ):

    

    # Global variables
    num_of_train_iterations = epochs
    # `number_of_layers` hidden layers with `hidden_dim` units each
    hidden_dim = 32
    number_of_layers=2

    learning_rate = pi_lr
    batch_size=steps_per_epoch
    discount_factor = gamma


    #init

    # init log
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    #make gym enviornment
    env = env_fn()
    obs_dim = env.observation_space.shape[0]


    # model - build computation graph
    with tf.variable_scope('model'):
        #input layer
        obs_ph = tf.placeholder(dtype=np.float32, shape=(None,obs_dim),name='obs_ph')
        #mlp (Multi Layer Perceptron) - hidden layers
        #define sizes for hidden layers
        hidden_sizes = [hidden_dim] * number_of_layers
        #build hidden layers
        x = obs_ph
        for h in hidden_sizes:
            x = tf.layers.dense(x, units=h,activation=tf.tanh)
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
        obs, reward, done, ep_rews = env.reset(), 0, False, []
        while len(batch_obs) < batch_size:
            #record batch
            # env.render()
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
                # y = discount_factor ** np.arange(n) 
                # rtgs_array = np.zeros_like(x, dtype=np.float32)
                # for step in range(n):
                #     rtgs_array[step] = sum(x[step:]*y[:n-step])
                rtgs_array  = np.zeros_like(x)
                for i in reversed(range(n)):
                    rtgs_array [i] = x[i] + (rtgs_array [i+1] if i+1 < n else 0)
                batch_rtgs += list(rtgs_array)
                #reset episodic data
                obs, reward, done, ep_rews = env.reset(), 0, False, []
                

        # TODO: normalize advs trick:
        # batch_advs = np.array(batch_rtgs)
        # batch_advs = (batch_advs - np.mean(batch_advs))/(np.std(batch_advs) + 1e-8)
        feed_dict={obs_ph: np.array(batch_obs[:len(batch_rtgs)]), acts_ph: np.array(batch_acts[:len(batch_rtgs)]), rtgs_ph: np.array(batch_rtgs)}
        batch_loss, _ = session.run([loss, optimizer], feed_dict=feed_dict)

        # log epoch results
        logger.log_tabular('Epoch', iter)
        logger.log_tabular('TotalEnvInteracts', (iter+1)*batch_size)
        logger.log_tabular('loss', batch_loss)
        logger.log_tabular('AverageEpRet', np.mean(batch_rets))
        logger.log_tabular('epispode mean length', np.mean(batch_lens))
        logger.dump_tabular()
    # end of simple_vpg



# just for debug . Experiments are done using commandline\script\experimentgrid
if __name__ == '__main__':
    simple_vpg()



