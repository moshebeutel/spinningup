import numpy as np
import tensorflow as tf
import gym
from spinup.utils.logx import EpochLogger



def simple_dqn(env_fn = lambda : gym.make('CartPole-v1')
    , actor_critic=None
    , ac_kwargs=dict()
    , seed=0
    , episodes_per_epoch=1000
    , epochs=1000
    , gamma=0.99
    , logger_kwargs=dict()
    , save_freq=1000
    , hidden_dim=32
    , n_layers=1
    , lr=1e-3
    , batch_size=32
    , target_update_freq=2500
    , final_epsilon=0.05
    , finish_decay=50000
    , replay_buffer_size=25000
    , steps_before_training=5000
    , n_test_eps = 10
    ):


    max_steps_per_epoch  = 5000

    # Global variables
    num_of_train_iterations = epochs
    # `number_of_layers` hidden layers with `hidden_dim` units each

    number_of_layers = n_layers
    learning_rate = lr
    discount_factor = gamma
    epsilon = 0.1
    init_epsilon = epsilon
 

    # init log
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    #make gym enviornment
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    number_of_actions = env.action_space.n

    #define evaluation network
    with tf.variable_scope('evaluation_network'):
        #input layer
        obs_ph = tf.placeholder(dtype=tf.float32, shape=(None,obs_dim), name='obs_ph')
        #mlp - #mlp (Multi Layer Perceptron) - hidden layers
        hidden_sizes = [hidden_dim] * number_of_layers
        x = obs_ph
        for h in hidden_sizes:
            x = tf.layers.dense(x, units=h, activation=tf.tanh)
        #output layer
        eval_net = tf.layers.dense(x,units=number_of_actions,activation=None)
    #define taget network
    with tf.variable_scope('target_network'):
        #input layer
        obs_target_ph = tf.placeholder(dtype=tf.float32, shape=(None,obs_dim), name='obs_target_ph')
        #mlp - #mlp (Multi Layer Perceptron) - hidden layers
        hidden_sizes = [hidden_dim] * number_of_layers
        x = obs_target_ph
        for h in hidden_sizes:
            x = tf.layers.dense(x, units=h, activation=tf.tanh)
        #output layer
        target_net = tf.layers.dense(x,units=number_of_actions,activation=None)


    #define loss function
    selected_action_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='selected_action_ph')
    reward_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='reward_ph')
    done_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='done_ph')
    actions_one_hot = tf.one_hot(selected_action_ph, number_of_actions)
    q_a = tf.reduce_sum(actions_one_hot * eval_net,axis=1)
    #use target network to approximate TD
    target = reward_ph + discount_factor * (1-done_ph) * tf.stop_gradient(tf.reduce_max(target_net, axis=1))
    loss = tf.reduce_mean((q_a - target)**2)

    #init replay buffer
    replay_current_obs = np.zeros([replay_buffer_size, obs_dim], dtype=np.int32)
    replay_next_obs = np.zeros([replay_buffer_size, obs_dim], dtype=np.int32)
    replay_selected_action = np.zeros(replay_buffer_size, dtype=np.int32)
    replay_reward =np.zeros(replay_buffer_size, dtype=np.float32)
    replay_done = np.zeros(replay_buffer_size, dtype=np.float32)

    # update op for target network
    main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluation_network')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
    assign_ops = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
    target_update_op = tf.group(*assign_ops)

    # define train optimizer_operation
    optimizer_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # init session
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    logger.setup_tf_saver(session, inputs={'x': obs_ph}, outputs={'q': eval_net})

    #TODO 
    total_number_of_steps = steps_before_training + epochs * max_steps_per_epoch
    current_index = replay_buffer_size - 1
    epoch = 0
    #reset epoch data
    epoch_rews, epoch_lens, epoch_losses,  epoch_qs = [], [], [], []
    #reset episodic data
    obs, reward, done, ep_rews, ep_len, episode_num, end_of_epoch = env.reset(), 0, False, 0, 0, 0, False
    last_number_steps = 0 
    for step in range(total_number_of_steps):
        #get action - 
        #     epsilon greedy
        selected_action = 0
        if np.random.rand() < epsilon :
            #exploration
            selected_action = np.random.randint(number_of_actions)
        else:
            #exploitation
            estimated_q = session.run(eval_net, feed_dict={obs_ph: obs.reshape(1,-1)})
            selected_action = np.argmax(estimated_q)
        
        # preform one step in gym enviornment
        #  receive observation reward and whether the episode has ended
        obs, reward, done, _  = env.step(selected_action)

        #store information in replay buffer
        #TODO deal with first and done
        replay_next_obs[current_index] = obs

        current_index = step % replay_buffer_size
        replay_current_obs[current_index] = obs
        replay_selected_action[current_index] = selected_action
        replay_reward[current_index] = reward
        replay_done[current_index] = done

        ep_rews += reward
        ep_len += 1

        if done:
            episode_num += 1
            #save episodic data 
            epoch_rews.append(ep_rews)
            epoch_lens.append(ep_len)
            #reset episodic data
            obs, reward, done, ep_rews, ep_len, end_of_epoch = env.reset(), 0, False, 0, 0, episode_num % episodes_per_epoch == 0
         
            
        #first `steps_before_training` do no train - replay buffer is too small
        if step > steps_before_training:
        #single train iteration
            #get data from replay
            trained_indices = np.random.randint(min(replay_buffer_size, step), size = batch_size)
            trained_observation = replay_current_obs[trained_indices]
            trained_next_observation = replay_next_obs[trained_indices]
            trained_selected_action = replay_selected_action[trained_indices]
            trained_reward = replay_reward[trained_indices]
            trained_done = replay_done[trained_indices]


            #if (step % save_freq == 0) or (step >= total_number_of_steps - 1):
            #    logger.save_state({'env': env}, None)
            # train eval network
            step_loss, curr_q, _  = session.run([loss, q_a, optimizer_operation], feed_dict={obs_ph: trained_observation, 
                                                                     obs_target_ph: trained_next_observation, 
                                                                     selected_action_ph: trained_selected_action,
                                                                     reward_ph: trained_reward,
                                                                     done_ph: trained_done})

            #just for logging
            epoch_losses.append(step_loss)
            epoch_qs.append(curr_q)
            
            
        
            if end_of_epoch:
                logger.save_state({'env': env}, None)
                # update target network
                session.run(target_update_op)
                
                epoch += 1
                
                #test epoch
                ep_rets, ep_lens = [], []
                for _ in range(n_test_eps):
                    obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                    while not(done):
                        #env.render()
                        estimated_q = session.run(eval_net, feed_dict={obs_ph: obs.reshape(1,-1)})
                        selected_action = np.argmax(estimated_q)
                        obs, rew, done, _ = env.step(selected_action)
                        ep_ret += rew
                        ep_len += 1
                    ep_rets.append(ep_ret)
                    ep_lens.append(ep_len)
                
                test_ep_ret =  np.mean(ep_rets)
                test_ep_len =  np.mean(ep_lens)

                obs, rew, done, ep_ret, ep_len, end_of_epoch = env.reset(), 0, False, 0, 0, False

                # log epoch results
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('TotalEnvInteracts', step - last_number_steps)
                logger.log_tabular('loss', np.mean(epoch_losses))
                logger.log_tabular('AverageEpRet', np.mean(test_ep_ret))
                logger.log_tabular('epispode mean length', np.mean(test_ep_len))
                logger.dump_tabular()

        
                epoch_rews, epoch_lens, epoch_losses,  epoch_qs, last_number_steps= [], [], [], [], step
                
            
                #adapt epsilon
                epsilon = 1 + (final_epsilon - 1)*min(1, step/finish_decay)

# just for debug . Experiments are done using commandline\script\experimentgrid
if __name__ == '__main__':
    simple_dqn()

