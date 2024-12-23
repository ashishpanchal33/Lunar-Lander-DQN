import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import deque
import random
import copy
import time
import pandas as pd
import os
from numpy import nanmean, nansum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from matplotlib import animation
import matplotlib.pyplot as plt
import imageio
from IPython.display import Image
import glob

base_address = "./model_data/"
address_modifier= "test_data_check/"
#pleasre refer bellow for definitions

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setting GPU as default
torch.cuda.set_device(0)



#neural network module ( only 2 hidden layers)
    #Input : 
    #-input_size, int (number of input features)
    #- hidden_layer_size , tuple (layer1,layer2), int
    # -output layer , int ( number of actions)
    # act, torch.nn   activaton function (default SELU)
    #seed : int , for libraries [random, numpy, torch] (default = 1)

class NN_model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, act=nn.SELU,seed = 1, rang = rang):
        super(NN_model, self).__init__()
        #print(input_size, hidden_layer_sizes, output_size)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        #print(torch.initial_seed(), torch.seed())
        self.linear_SELU = nn.Sequential(
            nn.Linear(input_size,hidden_layer_sizes[0]),
            act(),
            nn.Linear(hidden_layer_sizes[0],hidden_layer_sizes[1]),
            act(),
            nn.Linear(hidden_layer_sizes[1],output_size),   
        )

    def forward(self, x):
        x = self.linear_SELU(x)
        return x

    

    
    
#deep learning model class :
    #Input
        #env , : Lunar landar environment ( required, positional )
        #gamma = 0.99, : float (+ve) (discount rate)
        #alpha = 0.0005, : float (+ve) (learning rate for the model
        #batch_size = 64, : int (+ve) number of records per learning itterations
        #epsilon_start = 1, : float (0,1] starting value of exploratory probablity, deciding eta-gready action with probablity (1-e)
        #epsilon_min = 0.01 : float (0,1] minimum possible value of after eta is decayed.
        #C = 3 : int +ve ( Skip rate frequency )
        #N = 1 : int +ve ( target DQN update freuency)
        #epsilon_end = 1000, : int +ve, if using step based decay , decides maximum number of steps to decay from epsilon start to epsilon min
        #epsilon_decay = 0.00005, : if using decay rate for reducing eta, used to run at every step
        #eps_dcay_lim = 0, : boolean ( 1, 0), decay method decidor, if 0 then use decay rate to update eta, if 1 use epsilon_end to update eta
        #epsilon_change_state =0, : boolean (1,0) if 1, then reduce epsilon to 33%, else use as it is  ( Not to be used)
        #replay_memory = 100000, : int (+ve) , size of replay memory
        #lambda_ = 1, : (0,1] , not being used
        #input_size=8, :  int >0, (number of input features)
        #hidden_layer_sizes=[256,128], : list | tuple (layer1,layer2)  neurons, int
        #output_size=4, : int ( number of actions)
        #act=nn.SELU, : torch.nn. activaton function (default SELU)
        #model_data_dir ="./model_data", : str, based address ( to store, models, gif and data , if not present, it is created.
        #address_modifier ="", : str, sub_directory under model_data_dir , if not present, it is created.
        #specific_path="", :"str, only to be used to reviving models, and enable training on existing models or run tests on existing model. if present =, then address_modifier and model_data_dir are not used
        #act_name ="SELU", : str,name of the activation function, used to create model internal folders in over all address
        #env_name="LunarLander_v2",  str, name of the environment, used to create model internal folders in over all address
        #addition_details ="", :additionaltions specifiers to be added in the file or internal forlder namer after act_name and _env_name
        #save_option = 1, : boolean, (0,1), if 1 then the training, testing and model data is saved. 
        #return_option =1, : boolean, (0,1), if 1 then the training function and test function share, training and test records (episode count, score, steps per episode, timr of processing 
        #active_model = None, : Torch.nn.module , or NN_model class instance,  can be fed to perform training and testing of existing model ( psi = 250, not a requirement though) 
        #goal_model =None, Torch.nn.module , or NN_model class instance,  can be fed to perform training and testing of existing model, ( psi = 200, not a requirement though) 
        #target_model =None,Torch.nn.module , or NN_model class instance,  can be fed to perform training of existing model, ( psi = 200, not a requirement though) 
        #target_update =True, Boolean, if True, target network learning is performed, if zero use active only network to bootstrap
        #learning_new =True, : Boolean, if Ture, the active model and target model are created with other provided information, else try to used provided models, if not found, it will create new models,  ( ref handle_models) function
        #super_goal =250, float, defines the end of training, represnted by active model
        #sub_goal =200, float, defined a subgoal. represnted by goal model, whil training , a copy f active model is done when the avg score per 100 consicutive episodes reached 200
        #seed = 1 , seed to initialized, model weights, environment state, actitions and random, numpy class.
        
        
    #Class function
        #- __Init__ :
                #initilize all provided parameters, and copy to self. including active and target models 
                # also set loss function as MSEloss
                #set optimizer.
                #returns nothing
        # handle_models: handle_models(self, active_model = None, goal_model =None,target_model =None, learning_new =True ):
            #internal function to initialize active, goal and target models. ( refer function for more understaning.
             # can be externally called to set class intance models separetly.
             #returns nothing
       #update_epsilon : self
            #internal function to handle epsilon updated, 
                #if eps_dcay_lim : 1, epsilon -= epsilon_decay
                #if eps_dcay_lim : 0, epsilon -= epsilon_delta ( calculated as (self.epsilon_start - self.epsilon_min)/self.epsilon_end))
                #returns nothing
       #choose_action: (self, state):
            # input, state vector
            #used epsilon greedy policy and active model to decide action
                # if random value between [0,1] < epsilon, take randome actions from the action_space
            #return action , as int
       #_learning:
                #no input,
                # runs one instance of DQN algorithm, by internally taking batchsize, replay memory, loss function, optimizer, active model, target model 
               # calculate delta weights by backpropagation and  update w.
               # here also it is decided to update target model with N upates of active model, stored in a param update logger.
       #run_experience_and_learn:
                # no input,
                # exploration and DQN triggering function,
                # calls, update_epsilon, choose action, learning function and pd_model_save function.
                # records the (st,at,rt+1,st+1) tuple and update the replay memory
                # terminates episodes, if episodes >6000 or score (which is sum of returns in an episode) < -800 or avg score /100 consecutive episode >=supergoal
                # calculates, avg score of 100 consicutive episodes, score per episode, steps per episode, avg steps per 100 consicutive episode, processing time per episode.
                #store goal model when the score reached subgoal level.
                # call pd_model_save to save the model training data and active , goal models
       #pd_model_save:
                #(self,pd_Data_df,file_name, goal_model_save = False, active_model_save = False  )
                # input pandas data frame and file_name, calculated as informed above, and ave the file on the path.
                # also saves the current active, goal models to the path, if set to true
                #returns nothing
       #run_tests(self, model = None, model_num = 1, episodes = 100):
                #Input, model : Torch.nn.module or NN_model class object for testing
                        # for externally running the training for a model
                #model_num : boolean ( 0,1) , if 1, use current active model for testing, if 0, use goal model for testing.
                #episodes : int (+ve) , number of episodes to test upon.
                ##define file name. as test_model_info + subgoal|supergoal|"" +.csv
                #model uses the internal environment of the class, by initilizing it to base seed +1 to produce differnt , actions  from environment, along with internal randomvalues in use.
                # the model is run on every episode, storing the score per episode, stime take per episode and steps per episode.
                # by calling pd_model_save, and internally passed address ( as defined above )
                    # the data is saved in the csv at the provided locations ( for more info please check the code.
         
        
                        
#External function :
        #  save_gif(model, env,location, count = 1,seed=1, save = True):  #####saves render gif
                    #model :torch.nn.module or NN_model instance (which is suitable for the environment)
                    #env : Open gym any suitable invironment for the model ( or vis-a-versa)
                            # for this problem : Lunar landar environment ( required, positional )
                    #location : to save the gif if any created
                    # count : int, +ve ,number of episode to be solved and renderd.
                    #seed = 1 : int,  if a specific speed is know for replication
                    #save = true  : Boolean , if true, then save the gif created of the episodes at 60 fps as, location+"/model_.gif"
        #get_data_from_csv(base_address = "./model_data/",address_modifier= ""): #####takes data from all the training, testing data files created and aggregates.
                    #base_address = "./model_data/" : str ( address ) , of the primary folder of the model data.
                    #address_modifier = "" : : str ( address ), if data moved or stored to any sublocation., address of sublocation
                    # internally loads all training_model_info files ,test_model_info_200 files , test_model_info_250 files.
                    #engineers follwing data for following files:
                       #1. training_model_info         
                        '''['episode_no', 'Score', 'step_count', 'exec_time', 'rolling_step_count',
                       'rolling_Score', 'model_identifier', 'identifier_param', 'loss_fn_env',
                       'model_name', 'reached_goal', 'more_than_200', 'for_goal_analysis',
                       'rolling_exec_time', 'step_count_cumsum', 'exec_time_cumsum']'''
                        '''
                       #test_model_info_200. test_model_info_250
                       
                       ['episode_no', 'Score', 'step_count', 'exec_time', 'model_identifier',
                       'identifier_param', 'loss_fn_env', 'model_name', 'reached_goal',
                           'Hovering']
                        
                        
                        '''
                
                        # saves the files in the primary location i.e "./"
                        #also return the files as pandas data frames
            
            #Auto_test_all_models( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*"):
                       # env = lunar landar env.
                       # base address, : base address of the model directory
                       #address_modifier : as defined above
                       #required, NN_model class to be declread before running
                       # auto identifies and collect all the active and goal models pickles in the locations
                       # create a deep learning model class
                       # provides, the model and specific address, and auto run the testing for 100 episodes and save the generated data in the model folder locations.
                        # also required the deep QN class
                    
          #Auto_gif_creator( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*")
                       #similar tot auto_test_all_models:
                       #auto_gif_creator:
                       # find and runs the model for all the addresses on an internal 'list:
                           #["base","C_*","_N_*","repeat_*","gamma_*","alpha_*","ep_decay_*","batch_size_*","state_space_*", "replay_memory_size_*"]
                      #internall calls save_gif function
                    # dependency , required save gif and NN_module definitions
                       
    
    
###################################
###### model training and data saving functions per parameters
# no input required, however are dependent on above methods
# following are all the methods, parameter as name 
            #base_test()
            #repeat_test()
            #gamma_test()
            #ep_decay()
            #alpha_test()
            #target_network()
            #skip_test()
            #neurons()
            #batch_size()
            #replay_memory()
     #####
        #general flow :
            #create a emply list for
                # models,
                #train_data
                #test goal 250 data
                #test goal 200 data
            # create a list of feature variants
                # examples : gamma_range = [1,0.99,0.8,0.5,0.4,0.3]
            #create list of argument dict, from the parameter list 
                # examples : dictionary_loop_gamma_test = [{'env' : gym.make('LunarLander-v2'),'gamma' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"gamma_"+str(i)} for i in gamma_range ]
            # loop over the list of DQN class parameters = :
                #create class and append it ot models list, by passing current parameters
                #run training on the model, # which saves model on the locations ,and training data at the location, and returns the data as well. the returned dataframe is appended in the train data list.
                # test for 200 model is run, data is saved and appended in the deinfed list
                # test for 250 is run, data is saved and appended in the defined list.
                
                #nothing else is returned, as the data is besing saved in the locations.
                # however the code for a perticular parameter can be modified to return the appended data.
                
##########################################################
#########################################################
     #Please note after the run of all the above functions data and model functions 
############## get_data_from_csv() is called to aggregate the data and create ananalytical data structions,
        # followed by a call to  -- data_handler() funtion to save anther part of the anlytical data and compare all the model train and test performances.
################################
#############  #data_handler():
                    # this reads the data created and saved by get_data_from_csv() functions:
            #for the time being the address of these saved files are marked as static, however these can be modified to by the user.
            # after creating analytical dataset, it saves 4 data set in the baed locations:
                #: table_for_analysis_250,table_for_analysis_200,table_for_analysis,table_for_analysis_goal
                # these files contains summary of all the models,, training as well as testing processed by the previous functions.
                # it also passes the read data from the previous functions:
                    #: training_model_info,test_model_info_200,test_model_info_250
######################################################
################################################
######### graph functions
    # following are the final graph ploting ans saving functions.
    # they use the data return from data_handler()
            #graph_1()
            #graph_2()
            #repeat_1()
            #compare_200_250()
            #repeat_2()
            #skip_test_1()
            #skip_test_2()
            #target_1()
            #target_2()
            #batch_test_1()
            #batch_test_2()
            #batch_test_3()
            #replay_memory_test_1()
            #replay_memory_test_2()
            #replay_memory_3()
    # there is no other inpur required for these functions, they use global variables which should be saved as the return from data_handler():
            #they save the graph plotted in the  primary folder "./"
            # additionlly the graph is also plotted on the console
    
#############################################
#steps:
    #1. define all the class and all the functions
    #2. call all data functions
            #base_test()
            #repeat_test()
            #gamma_test()
            #ep_decay()
            #alpha_test()
            #target_network()
            #skip_test()
            #neurons()
            #batch_size()
            #replay_memory()
            
            
    #3.. call ADS creator functions:
            #training_model_info, test_model_info_200,test_model_info_250 = get_data_from_csv(address_modifier=address_modifier)

            #training_model_info,test_model_info_200,test_model_info_250,table_for_analysis_250,table_for_analysis_200,table_for_analysis,table_for_analysis_goal = data_handler()    
            
    #4.optionally call :
        # Auto_gif_creator( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*")
        #  Auto_test_all_models( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*")
    
    #5. call all the plot functions:
            #graph_1()
            #graph_2()
            #repeat_1()
            #compare_200_250()
            #repeat_2()
            #skip_test_1()
            #skip_test_2()
            #target_1()
            #target_2()
            #batch_test_1()
            #batch_test_2()
            #batch_test_3()
            #replay_memory_test_1()
            #replay_memory_test_2()
            #replay_memory_3()
        
  
            
#for running all: call : run_all()
            # may take very long time based on the processesing power of the system
    
    
    
    
    
    
    
    
    
    
    
    
class Deep_Q_Network3:
    def __init__(self, env , gamma = 0.99,alpha = 0.0005,batch_size = 64, 
                 epsilon_start = 1,epsilon_min = 0.01,C = 3, N = 1, epsilon_end = 1000,epsilon_decay = 0.00005,eps_dcay_lim = 0,
                 epsilon_change_state =0, replay_memory = 100000,
                 lambda_ = 1, input_size=8, hidden_layer_sizes=[256,128], output_size=4,
                 act=nn.SELU, model_data_dir ="./model_data",address_modifier ="",specific_path="",
                 act_name ="SELU",env_name="LunarLander_v2",addition_details ="", save_option = 1, return_option =1,
                active_model = None, goal_model =None,target_model =None, target_update =True, learning_new =True,
                 super_goal =250, sub_goal =200, seed = 1
                 
                ):
        
        
        
        
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        
        
        
        
        # init environment
        self.env = env
        self.env_name=env_name
        self.act_name=act_name
        
        if (specific_path != "") and os.path.exists(specific_path):
            self.dir_name = specific_path
        else:
            self.model_data_dir = model_data_dir+address_modifier
            self.addition_details = addition_details
            if not os.path.exists(model_data_dir):
                os.mkdir(model_data_dir)
            if not os.path.exists(self.model_data_dir):
                os.mkdir(self.model_data_dir)
            #self.dir_name = self.model_data_dir+'./'+self.act_name+"_"+self.env_name+"_"+self.addition_details
            self.dir_name = self.model_data_dir+'/'+self.act_name+"_"+self.env_name+"_"+self.addition_details
        
        
        self.save_option = save_option
        self.return_option = return_option 

        #experience replay buffer
        self.replay_memory  = deque(maxlen= replay_memory) #int(1e5))
        # discount factor
        self.gamma = gamma
        #learning rate
        self.alpha = alpha
        self.epsilon_change_state =epsilon_change_state
        if (self.epsilon_change_state ==1):
            self.epsilon_decay = epsilon_decay*0.3
        else:
            self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start        
        self.epsilon_min = epsilon_min
        self.epsilon_end = epsilon_end
        self.eps_dcay_lim = eps_dcay_lim
        self.epsilon_delta = (self.epsilon_start - self.epsilon_min)/self.epsilon_end
        self.lambda_ = lambda_
        # frequency with which target network is updated.
        self.C = C
        
        self.N = N
        
        
        # number of experience tuples used in computing the gradient descent parameter update.
        self.batch_size= batch_size
        self.episode_counter =0
        self.param_update_counter =0
        self.state_counter =0
        self.target_update = target_update
        self.super_goal =super_goal
        self.sub_goal =sub_goal
        
        self.input_size=input_size 
        self.hidden_layer_sizes=hidden_layer_sizes
        self.output_size=output_size
        self.act = act

        
        
        #handling input models, to continue training and testing of any existing model
        #usage of target model during learning is subject to "target_update" flag 
                
        self.handle_models(active_model = active_model, goal_model =goal_model,target_model =target_model, learning_new =learning_new )
        
        self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.active_model.parameters(),lr=self.alpha)        
        
        
    def handle_models(self, active_model = None, goal_model =None,target_model =None, learning_new =True ):
        
        if(learning_new ==True):
            self.active_model = NN_model(input_size=self.input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                     output_size=self.output_size, act=self.act,seed = self.seed)

            self.target_model = NN_model(input_size=self.input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                 output_size=self.output_size, act=self.act,seed = self.seed)             
        else:
            if(active_model == None):
                if(target_model == None):
                    if(goal_model == None):
                        self.active_model = NN_model(input_size=self.input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                                 output_size=self.output_size, act=self.act,seed = self.seed)

                        self.target_model = NN_model(input_size=self.input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                             output_size=self.output_size, act=self.act) 
                        self.goal_model = NN_model(input_size=self.input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                             output_size=self.output_size, act=self.act,seed = self.seed) 

                    else:
                        self.active_model = copy.deepcopy(goal_model)
                        self.target_model = copy.deepcopy(goal_model)
                        self.goal_model = goal_model


                else:
                    self.active_model = copy.deepcopy(target_model)
                    self.target_model = target_model

                    if(goal_model == None):
                        self.goal_model = copy.deepcopy(target_model)
                    else:
                        self.goal_model = goal_model
            else:
                self.active_model = active_model

                if(target_model == None):
                    self.target_model = copy.deepcopy(active_model)
                else:
                    self.target_model = target_model 



                if(goal_model == None):
                    self.goal_model = copy.deepcopy(active_model)
                else:
                    self.goal_model = goal_model
        
        
    def update_epsilon(self):
        if self.eps_dcay_lim == 1:
            if self.episode_counter < self.epsilon_end:
                self.epsilon -= self.epsilon_delta
            else:
                self.epsilon = self.epsilon_min
        else:
            if self.epsilon > self.epsilon_min:
                #print(self.epsilon,self.epsilon_min,self.epsilon_decay )
                self.epsilon -= self.epsilon_decay 
            else:
                self.epsilon = self.epsilon_min        
        
    def choose_action(self, state):
        if np.random.random(1)[0] < self.epsilon:
            action  = self.env.action_space.sample()
        else:
            self.active_model.eval()
            with torch.no_grad():
                action_values = self.active_model(torch.Tensor(state))
            self.active_model.train()

            # choose the greedy action
            action = (action_values.argmax().item()) 
            
        return(action)
        
        
        
    def _learning(self):
        
        #update active model
        states, targets_forecast =[],[]
        
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        
        state, action, reward, next_state, done = (torch.Tensor(x) for x in zip(*mini_batch))
        
        
        if(self.target_update == True):
            Q_t1 = self.target_model(next_state).detach().max(dim=1)[0]
        else:
            Q_t1 = self.active_model(next_state).detach().max(dim=1)[0]
        
        
        # compute the new Q' values using the Q-learning formula
        Q_t_prime = reward + (self.gamma*Q_t1*(1-done))
        
        # get expected Q values from local model
        action_number = (action.long().unsqueeze(dim=1))
        expected_Q_a = (self.active_model(state).gather(dim=1, index=action_number))
        # compute the mean squared loss
        loss = self.loss(Q_t_prime.unsqueeze(dim=1),expected_Q_a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.param_update_counter +=1
        if((self.target_update == True) and (self.param_update_counter % self.N == 0)):
            self.param_update_counter = 0
            self.target_model.load_state_dict(self.active_model.state_dict())
        #self.target_model == copy.deepcopy(self.active_model)
        
        #adjusting epsilon
        if (self.epsilon_change_state ==0):
            self.update_epsilon()
                    
        
        #update target model
        
        
    def run_experience_and_learn(self):
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        
        
        
        
        
        
        
        #replay memory is already inisitalized
        #action value function Q with random weights is taken care by the model
        
        avg_score = 0
        score_list = []
        step_count_list =[]
        execution_time_list =[]
        #epsilon_list =[]
        score_que = deque(maxlen = 100)
        step_count = 0
        
        reached_goal_1_flag = 0
        reached_goal_1_details =[]
        
        
        while(avg_score <self.super_goal):
            
            if (reached_goal_1_flag ==0) and (avg_score >=self.sub_goal):
                self.goal_model = copy.deepcopy(self.active_model)
                reached_goal_1_flag = 1
                reached_goal_1_details = [self.episode_counter, avg_score]
            
            
            score = 0
            current_state = self.env.reset() 
            
            self.episode_counter +=1
            

            episode_step_count = 0
            
            start_time = time.time()
            #current_state = np.append(current_state,[episode_step_count])
            while(True):
                    
                #choosing action
                action = self.choose_action(current_state)
                
                step_count +=1
                episode_step_count+=1                    

                next_state, reward, done, _ = self.env.step(action)
                #next_state = np.append(next_state,[episode_step_count])
                score += reward
                #reward += - (0.001*episode_step_count)**2
                
                
                #appending to the memory
                self.replay_memory.append([current_state,action,reward,next_state,done])

                current_state = next_state
                #skipping experiences
                if (len(self.replay_memory)> self.batch_size) and (step_count%self.C ==0):
                    self._learning()
                
                if (self.epsilon_change_state ==1):
                    self.update_epsilon()
                    
                if done:# or episode_step_count >400:
                    break
                

                        
                
            score_list.append(score)
            step_count_list.append(episode_step_count)
            execution_time_list.append(start_time-time.time())
            score_que.append(score)
            if(len(score_que)==100):
                avg_score = np.mean(score_que)
            if(self.episode_counter %100 ==0):
                print("episode: ",  self.episode_counter, "avg_score :" , avg_score , " epsilon :", self.epsilon )
            
            if((self.episode_counter >6000) or (avg_score <-800) ):
                print("not able to find near optimal policy")
                if (reached_goal_1_flag ==0):
                    self.goal_model = copy.deepcopy(self.active_model)
                    reached_goal_1_flag = 1             

                break
        
        print('Complete')
        #env.render()
        #env.close()
       
        pd_Data_df = pd.DataFrame({"Score": score_list,"step_count":step_count_list,"exec_time": execution_time_list})
        pd_Data_df["rolling_step_count"] =pd_Data_df.step_count.rolling(100).mean()
        pd_Data_df["rolling_Score"] =pd_Data_df.Score.rolling(100).mean()
        
        if(self.save_option ==1):
            self.pd_model_save(pd_Data_df=pd_Data_df,file_name="training_model_info.csv", goal_model_save = True, active_model_save = True  )
        
        if(self.return_option ==1):
            return(pd_Data_df)
    
    
    def pd_model_save(self,pd_Data_df,file_name, goal_model_save = False, active_model_save = False  ):
            if not os.path.exists(self.dir_name):
                os.mkdir(self.dir_name)
            
            pd_Data_df.to_csv(os.path.join(self.dir_name,file_name)) 
            
            if(active_model_save):
                torch.save(self.active_model, os.path.join(self.dir_name,"active_model"))
            if(goal_model_save):
                torch.save(self.goal_model, os.path.join(self.dir_name,"goal_model"))                
    
    
    def run_tests(self, model = None, model_num = 1, episodes = 100):
        torch.manual_seed(self.seed+1)
        torch.cuda.manual_seed(self.seed+1)
        torch.cuda.manual_seed_all(self.seed+1)
        np.random.seed(self.seed+1)
        random.seed(self.seed+1)
        self.env.seed(self.seed+1)
        self.env.action_space.seed(self.seed+1)
        
        if (model_num == 1):
            model =self.active_model
            file_name = 'test_model_info_'+str(self.super_goal)+'.csv'
        elif model_num == 0:
            model = self.goal_model
            file_name = 'test_model_info_'+str(self.sub_goal)+'.csv'
        else:
            file_name = "test_model_info.csv"
            
        
        score_list = []
        step_count_list =[]
        execution_time_list =[]
        
        # since the backpropagation is not used and optimzer does not take steps , the model is not learning here
        for i_episode in range(episodes):
            observation = self.env.reset()
            score = 0
            
            episode_step_count = 0
            
            start_time = time.time()
            
            model.eval()

            for t in range(2000):
                with torch.no_grad():
                    action = model(torch.Tensor(observation)).detach().argmax().numpy()
                observation, reward, done, info = self.env.step(action)
                score += reward

                episode_step_count += 1

                if done:
                    break    
        
            score_list.append(score)
            step_count_list.append(episode_step_count)
            execution_time_list.append(start_time-time.time())
            
        pd_Data_df = pd.DataFrame({"Score": score_list,"step_count":step_count_list,"exec_time": execution_time_list})
        
        if(self.save_option ==1):
            self.pd_model_save(pd_Data_df=pd_Data_df,file_name=file_name, goal_model_save = False, active_model_save = False  )
        
        if(self.return_option ==1):
            return(pd_Data_df)
        
        
        
def save_gif(model, env,location, count = 1,seed=1, save = True):
    frames_RELU = []
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    
    for i_episode in range(count):
        observation = env.reset()
        score = 0
        for t in range(2000):
            frames_RELU.append(env.render(mode = 'rgb_array'))
            action = model(torch.Tensor(observation)).detach().argmax().numpy()
            observation, reward, done, info = env.step(action)
            score+=reward
            if done:
                print("Reward: ",score, "timesteps",(t+1), "model", location[13:])
                break
    env.close()
    if save:
        imageio.mimsave(location+"/model_.gif", frames_RELU,fps =60)        
        
        

        
def get_data_from_csv(base_address = "./model_data/",address_modifier= ""):
    address_modifier = address_modifier+"*"
    training_model_info = pd.DataFrame(columns=['Unnamed: 0','Score','step_count','exec_time','rolling_step_count','rolling_Score','model_identifier','identifier_param','loss_fn_env','model_name'])
    test_model_info_200 = pd.DataFrame(columns=['Unnamed: 0','Score','step_count','exec_time','model_identifier','identifier_param','loss_fn_env','model_name'])
    test_model_info_250 = pd.DataFrame(columns=['Unnamed: 0','Score','step_count','exec_time','model_identifier','identifier_param','loss_fn_env','model_name'])
    
    experiment_group_list = ["base","C_*","_N_*",
                             "repeat_*","gamma_*","ep_end_*","alpha_*","ep_decay_*","batch_size_*","state_space_*","replay_memory_size_*"]
    
    
    
    if not os.path.exists(base_address+address_modifier[:-1]):
        os.mkdir(base_address+address_modifier[:-1])
    
    
    for i in experiment_group_list:
        k = glob.glob(base_address+address_modifier+i)
        for j in k:

            model_identifier = (i[:-2] if(i!= "base") else "base")
            identifier_param = j[j.find(i[:-1])+len(i[:-1]):]
            loss_fn_env = j[len(base_address+address_modifier)-1:j.find(i[:-1])-1]
            mode_name = j[len(base_address+address_modifier)-1:]
            #print(model_identifier,identifier_param,over_all_model)

            data_model_info = pd.read_csv(j+"/training_model_info.csv")
            data_model_info["model_identifier"] =model_identifier
            data_model_info["identifier_param"]=identifier_param
            data_model_info["loss_fn_env"] = loss_fn_env
            data_model_info["model_name"] = mode_name
            
            
            data_model_200 = pd.read_csv(j+"/test_model_info_200.csv")
            data_model_200["model_identifier"] =model_identifier
            data_model_200["identifier_param"]=identifier_param
            data_model_200["loss_fn_env"] = loss_fn_env
            data_model_200["model_name"] =mode_name
            
            
            data_model_250 = pd.read_csv(j+"/test_model_info_250.csv")
            data_model_250["model_identifier"] =model_identifier
            data_model_250["identifier_param"]=identifier_param
            data_model_250["loss_fn_env"] = loss_fn_env
            data_model_250["model_name"] = mode_name
            

            training_model_info = training_model_info.append(data_model_info,ignore_index =True)
            test_model_info_200 = test_model_info_200.append(data_model_200,ignore_index =True)
            test_model_info_250 = test_model_info_250.append(data_model_250,ignore_index =True)
            #if(num >2):
            #    break
    training_model_info.columns=list(["episode_no"]) +list(training_model_info.columns[1:])
    test_model_info_200.columns=list(["episode_no"]) +list(test_model_info_200.columns[1:])
    test_model_info_250.columns=list(["episode_no"]) +list(test_model_info_250.columns[1:])
    
    
    training_model_info["reached_goal"] = np.nan
    training_model_info["more_than_200"] = training_model_info["rolling_Score"]>200

    training_model_info.loc[training_model_info[training_model_info["more_than_200"]
                       ].groupby(["model_name"]).apply(lambda x: x["rolling_Score"].idxmin()).values,"reached_goal"] =True

    training_model_info["for_goal_analysis"] =  training_model_info.apply(lambda x : (x["reached_goal"] == True) or (x["more_than_200"] == False), axis =1)
    test_model_info_200["reached_goal"] = (test_model_info_200["Score"]>=200).apply(lambda x: int(x))
    test_model_info_250["reached_goal"] = (test_model_info_250["Score"]>=200).apply(lambda x: int(x))
    test_model_info_200["Hovering"] = (test_model_info_200["step_count"]==1000).apply(lambda x: int(x))
    test_model_info_250["Hovering"] = (test_model_info_250["step_count"]==1000).apply(lambda x: int(x))
    training_model_info['exec_time'] = -1*training_model_info['exec_time']
    
    training_model_info["rolling_exec_time"] = training_model_info.groupby("model_name").exec_time.rolling(100).mean().values
    
    
    
    training_model_info.to_csv(base_address+address_modifier[:-1]+"training_model_info_all_models.csv")
    test_model_info_200.to_csv(base_address+address_modifier[:-1]+"test_model_info_200_all_models.csv")
    test_model_info_250.to_csv(base_address+address_modifier[:-1]+"test_model_info_250_all_models.csv")
    
    return(training_model_info, test_model_info_200,test_model_info_250)




def Auto_test_all_models( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*"):
    #env = gym.make('LunarLander-v2')
    #address_modifier = "test_data_check/*"
    k = glob.glob(base_address+address_modifier)
    for j in k:
        print(j)
        active_model = torch.load(j+"/active_model")
        goal_model = torch.load(j+"/goal_model")

        param_dict = {"env":env,"specific_path":j,
                     'active_model' : active_model,'return_option' :0, 'goal_model' :goal_model, 'target_update' :True, 'learning_new' :False,
                      'seed' : 2
                     }

        dqn = Deep_Q_Network3(**param_dict)
        dqn.run_tests(model_num = 1)
        dqn.run_tests(model_num = 0)    


        
def Auto_gif_creator( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*"):
    experiment_group_list = ["base","C_*","_N_*",
                             "repeat_*","gamma_*","alpha_*","ep_decay_*","batch_size_*","state_space_*", "replay_memory_size_*"]

    for i in experiment_group_list:
        k = glob.glob(base_address+address_modifier+i)
        for j in k:
            revived_model = torch.load(j+"/active_model")
            save_gif(revived_model, env,j)        
        
        
        
        
        
        
        
        
#### model training and data saving    



####################comparing differnt activation function
def base_test():
    dictionary_loop = [{'env' : gym.make('LunarLander-v2'), 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"base"},
                       {'env' : gym.make('LunarLander-v2'), 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.ReLU,'act_name' :"ReLU",'env_name':"LunarLander_v2",'addition_details':"base"},
                       {'env' : gym.make('LunarLander-v2'), 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.GELU,'act_name' :"GELU",'env_name':"LunarLander_v2",'addition_details':"base"}
                      ]


    model_list = []
    train_data =[]
    test_data_250 =[]
    test_data_200 =[]

    for n,i in enumerate(dictionary_loop):
            model_list.append(Deep_Q_Network3(**i))
            train_data.append(model_list[n].run_experience_and_learn())
            test_data_250.append(model_list[n].run_tests(model_num = 1))
            test_data_200.append(model_list[n].run_tests(model_num = 0))




################

##################################
###extensively testing LeakyRLU


def repeat_test():

    # Random seed repeat test
    model_list_repeat_test = []
    train_data_repeat_test  =[]
    test_data_250_repeat_test  =[]
    test_data_200_repeat_test  =[]

    repeat_range = range(10)


    dictionary_loop_repeat_test = [{'env' : gym.make('LunarLander-v2'), 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"repeat_"+str(i),"seed":i}
                                  for i in repeat_range
                      ]


    for n,i in enumerate(dictionary_loop_repeat_test):

        model_list_repeat_test.append(Deep_Q_Network3(**i))
        train_data_repeat_test.append(model_list_repeat_test[n].run_experience_and_learn())
        test_data_250_repeat_test.append(model_list_repeat_test[n].run_tests(model_num = 1))
        test_data_200_repeat_test.append(model_list_repeat_test[n].run_tests(model_num = 0))


#####Gamma test###########
def gamma_test():

    model_list_gamma_test = []
    train_data_gamma_test  =[]
    test_data_250_gamma_test  =[]
    test_data_200_gamma_test  =[]

    gamma_range = [1,0.99,0.8,0.5,0.4,0.3]


    dictionary_loop_gamma_test = [{'env' : gym.make('LunarLander-v2'),'gamma' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"gamma_"+str(i)}
                                  for i in gamma_range
                      ]



    for n,i in enumerate(dictionary_loop_gamma_test):

        model_list_gamma_test.append(Deep_Q_Network3(**i))
        train_data_gamma_test.append(model_list_gamma_test[n].run_experience_and_learn())
        test_data_250_gamma_test.append(model_list_gamma_test[n].run_tests(model_num = 1))
        test_data_200_gamma_test.append(model_list_gamma_test[n].run_tests(model_num = 0))




###################################
#epsilon decay test 
def ep_decay():
    model_list_delimma_test = []
    train_data_delimma_test  =[]
    test_data_250_delimma_test  =[]
    test_data_200_delimma_test  =[]

    #epsilon_decay_range = [0.0001,0.00005,0.00001,0.000005]
    epsilon_decay_range = [0.0001,0.00001,0.000005]
    dictionary_loop_delimma_test = [{'env' : gym.make('LunarLander-v2'),'epsilon_decay' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"ep_decay_"+str(i)}
                                  for i in epsilon_decay_range
                      ]


    for n,i in enumerate(dictionary_loop_delimma_test):

        model_list_delimma_test.append(Deep_Q_Network3(**i))
        train_data_delimma_test.append(model_list_delimma_test[n].run_experience_and_learn())
        test_data_250_delimma_test.append(model_list_delimma_test[n].run_tests(model_num = 1))
        test_data_200_delimma_test.append(model_list_delimma_test[n].run_tests(model_num = 0))


#############################################
##learning rate_alpha test
def alpha_test():
    model_list_learning_rate_test = []
    train_data_learning_rate_test  =[]
    test_data_250_learning_rate_test  =[]
    test_data_200_learning_rate_test  =[]

    alpha_range = [0.01, 0.005, 0.001,0.0005,0.0001]

    dictionary_loop_learning_rate_test = [{'env' : gym.make('LunarLander-v2'),'alpha' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"alpha_"+str(i)}
                                  for i in alpha_range
                      ]


    for n,i in enumerate(dictionary_loop_learning_rate_test):

        model_list_learning_rate_test.append(Deep_Q_Network3(**i))
        train_data_learning_rate_test.append(model_list_learning_rate_test[n].run_experience_and_learn())
        test_data_250_learning_rate_test.append(model_list_learning_rate_test[n].run_tests(model_num = 1))
        test_data_200_learning_rate_test.append(model_list_learning_rate_test[n].run_tests(model_num = 0))
    
    
    
    
    
########################################################
####target network update frequency test 
def target_network():    
    model_list_update_rate_test = []
    train_data_update_rate_test  =[]
    test_data_250_update_rate_test  =[]
    test_data_200_update_rate_test  =[]

    N_range = [1,3,10,50,100,500,1000,5000]

    #setting update the per action
    dictionary_loop_update_rate_test = [{'env' : gym.make('LunarLander-v2'),"N": i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"_N_"+str(i)}
                                  for i in N_range
                      ]


    for n,i in enumerate(dictionary_loop_update_rate_test):

        model_list_update_rate_test.append(Deep_Q_Network3(**i))
        train_data_update_rate_test.append(model_list_update_rate_test[n].run_experience_and_learn())
        test_data_250_update_rate_test.append(model_list_update_rate_test[n].run_tests(model_num = 1))
        test_data_200_update_rate_test.append(model_list_update_rate_test[n].run_tests(model_num = 0))    

    
    
    
    
    
#######################################################
###########Skip step test
    
def skip_test():    
    model_list_skip_rate_test = []
    train_data_skip_rate_test  =[]
    test_data_250_skip_rate_test  =[]
    test_data_200_skip_rate_test  =[]

    C_range = [1,3,10,50,100]

    #setting update the per action
    dictionary_loop_skip_rate_test = [{'env' : gym.make('LunarLander-v2'),'C' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"C_"+str(i)}
                                  for i in C_range
                      ]


    for n,i in enumerate(dictionary_loop_skip_rate_test):

        model_list_skip_rate_test.append(Deep_Q_Network3(**i))
        train_data_skip_rate_test.append(model_list_skip_rate_test[n].run_experience_and_learn())
        test_data_250_skip_rate_test.append(model_list_skip_rate_test[n].run_tests(model_num = 1))
        test_data_200_skip_rate_test.append(model_list_skip_rate_test[n].run_tests(model_num = 0))    
    
    
############################################################3
############### neurons count in layers tets
    
def neurons():    
    model_list_state_space_test = []
    train_data_state_space_test = []
    test_data_250_state_space_test = []
    test_data_200_state_space_test = []

    neuron_range = [32,64,128,256]

    dictionary_loop_state_space_test = [{'env' : gym.make('LunarLander-v2'), 'input_size':8,'hidden_layer_sizes':[i,i],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"state_space_"+str(i)}
                                  for i in neuron_range
                      ]


    for n,i in enumerate(dictionary_loop_state_space_test):

        model_list_state_space_test.append(Deep_Q_Network3(**i))
        train_data_state_space_test.append(model_list_state_space_test[n].run_experience_and_learn())
        test_data_250_state_space_test.append(model_list_state_space_test[n].run_tests(model_num = 1))
        test_data_200_state_space_test.append(model_list_state_space_test[n].run_tests(model_num = 0))


    
    
    
    
########################################################################
# Batch size test 

def batch_size():

    model_list_batch_test = []
    train_data_batch_test = []
    test_data_250_batch_test = []
    test_data_200_batch_test = []

    batch_size_range = [8,32,64,128,256]

    dictionary_loop_batch_test = [{'env' : gym.make('LunarLander-v2'),'batch_size' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"batch_size_"+str(i)}
                                  for i in batch_size_range
                      ]


    for n,i in enumerate(dictionary_loop_batch_test):

        model_list_batch_test.append(Deep_Q_Network3(**i))
        train_data_batch_test.append(model_list_batch_test[n].run_experience_and_learn())
        test_data_250_batch_test.append(model_list_batch_test[n].run_tests(model_num = 1))
        test_data_200_batch_test.append(model_list_batch_test[n].run_tests(model_num = 0))



##########################################################################
###Replay memory 
def replay_memory():
    model_list_replay_memory_test = []
    train_data_replay_memory_test = []
    test_data_250_replay_memory_test = []
    test_data_200_replay_memory_test = []

    replay_memory_size_range = [5000,10000,50000,100000,500000]

    dictionary_loop_replay_memory_test = [{'env' : gym.make('LunarLander-v2'),'replay_memory' : i, 'input_size':8,'hidden_layer_sizes':[128,128],'output_size':4,
                     'act':nn.LeakyReLU,'act_name' :"LeakyReLU",'env_name':"LunarLander_v2",'addition_details':"replay_memory_size_"+str(i)}
                                  for i in replay_memory_size_range
                      ]


    for n,i in enumerate(dictionary_loop_replay_memory_test):

        model_list_replay_memory_test.append(Deep_Q_Network3(**i))
        train_data_replay_memory_test.append(model_list_replay_memory_test[n].run_experience_and_learn())
        test_data_250_replay_memory_test.append(model_list_replay_memory_test[n].run_tests(model_num = 1))
        test_data_200_replay_memory_test.append(model_list_replay_memory_test[n].run_tests(model_num = 0))


#####################################################################




def data_handler():

    training_model_info, test_model_info_200,test_model_info_250 = (pd.read_csv(base_address+address_modifier+"training_model_info_all_models.csv"),
                                                                    pd.read_csv(base_address+address_modifier+"test_model_info_200_all_models.csv"),
                                                                    pd.read_csv(base_address+address_modifier+"test_model_info_250_all_models.csv"))



    training_model_info.drop("Unnamed: 0",inplace = True,axis =1)
    test_model_info_200.drop("Unnamed: 0",inplace = True,axis =1)
    test_model_info_250.drop("Unnamed: 0",inplace = True,axis =1)









    training_model_info["step_count_cumsum"]= training_model_info.sort_values(['model_name','episode_no'], ascending = [True,True]).groupby(['model_name'])['step_count'].cumsum()
    training_model_info["exec_time_cumsum"]= training_model_info.sort_values(['model_name','episode_no'], ascending = [True,True]).groupby(['model_name'])['exec_time'].cumsum()


    table_for_analysis = training_model_info.iloc[training_model_info.groupby(["model_identifier",'identifier_param']).episode_no.idxmax(),].loc[training_model_info["model_identifier"] != "base",]
    table_for_analysis.loc[:,'identifier_param']= table_for_analysis['identifier_param'].astype("float64")
    table_for_analysis_grouped = table_for_analysis.groupby("model_identifier")



    table_for_analysis_goal = training_model_info.loc[training_model_info["reached_goal"] == True, ].loc[training_model_info["model_identifier"] != "base",]
    table_for_analysis_goal.loc[:,'identifier_param']= table_for_analysis_goal['identifier_param'].astype("float64")
    table_for_analysis_goal_grouped = table_for_analysis_goal.groupby("model_identifier")




    table_for_analysis_250_0 = test_model_info_250.groupby(["model_name",'loss_fn_env','identifier_param','model_identifier']).agg({
        'Score':(nanmean,'std'),'step_count' : (nanmean,'std') ,'exec_time' : (nanmean,'std') ,
        'reached_goal': 'sum','Hovering': 'sum'
    }).reset_index()

    table_for_analysis_250 = table_for_analysis_250_0.loc[table_for_analysis_250_0["model_identifier"] != "base",]



    table_for_analysis_250.loc[:,'identifier_param']= table_for_analysis_250['identifier_param'].astype("float64")
    table_for_analysis_250_grouped = table_for_analysis_250.groupby("model_identifier")


    table_for_analysis_200_0 = test_model_info_200.groupby(["model_name",'loss_fn_env','identifier_param','model_identifier']).agg({
        'Score':(nanmean,'std'),'step_count' : (nanmean,'std') ,'exec_time' : (nanmean,'std') ,
        'reached_goal': 'sum','Hovering': 'sum'
    }).reset_index()

    table_for_analysis_200 = table_for_analysis_200_0.loc[table_for_analysis_200_0["model_identifier"] != "base",]



    table_for_analysis_200.loc[:,'identifier_param']= table_for_analysis_200['identifier_param'].astype("float64")
    table_for_analysis_200_grouped = table_for_analysis_200.groupby("model_identifier")



    #for name,group in table_for_analysis_250_grouped:
    #    display(group.sort_values(["identifier_param",('Score',"nanmean")], ascending= [True,False]))


    table_for_analysis_250.to_csv(base_address+address_modifier+"test_250_analytical_overview.csv")
    table_for_analysis_200.to_csv(base_address+address_modifier+"test_200_analytical_overview.csv")
    table_for_analysis.to_csv(base_address+address_modifier+"train_end_analytical_overview.csv")
    table_for_analysis_goal.to_csv(base_address+address_modifier+"train_200_analytical_overview.csv")



    for name,group in table_for_analysis_grouped:
        print(name,"training_200")
        display(table_for_analysis_goal_grouped.get_group(name).sort_values(["identifier_param","episode_no",'rolling_Score'], ascending= [True,True,False]))

        print(name,"training_end")
        display(group.sort_values(["identifier_param","episode_no",'rolling_Score'], ascending= [True,True,False]))
        print("test_250")
        display(table_for_analysis_250_grouped.get_group(name).sort_values(["identifier_param",('Score',"nanmean")], ascending= [True,False]))
        print("test_200")
        display(table_for_analysis_200_grouped.get_group(name).sort_values(["identifier_param",('Score',"nanmean")], ascending= [True,False]))
    
    
    return(training_model_info,test_model_info_200,test_model_info_250,table_for_analysis_250,table_for_analysis_200,table_for_analysis,table_for_analysis_goal)

#training_model_info,test_model_info_200,test_model_info_250,table_for_analysis_250,table_for_analysis_200,table_for_analysis,table_for_analysis_goal = data_handler()  
#faster learning, low episode count,
#smoother learning = low deviation during training
# good performance low step count, low avg score,low failures and low deviation





def graph_1():
        #color_dict_graph1 = {'rolling100_Score': '#CB4335', 'Score': '#F2D7D5','step_count' : '#EBF5FB','rolling100_step_count' : '#85C1E9'}
    color_dict_graph1 = {'rolling100_Score': '#CB4335', 'Score': '#F2D7D5','step_count' : '#BEE1FB','rolling100_step_count' : '#0474C5'}
    graph1_df = training_model_info[training_model_info["model_name"] == "LeakyReLU_LunarLander_v2_base"][['episode_no', 'Score', 'step_count','rolling_step_count',
           'rolling_Score']]

    graph1_df.columns = ['episode_no', 'Score', 'step_count', 'rolling100_step_count','rolling100_Score']
    plt.rcParams.update({'font.size': 10})
    plt.figure()


    ax_graph1_step = graph1_df[['episode_no', 'step_count','rolling100_step_count']].plot(x='episode_no', rot=0, 
                              color=[color_dict_graph1.get(x, '#333333') for x in['step_count','rolling100_step_count']],
                              kind='line',style ='-',fontsize=14,figsize=(8,5) )
    ax_graph1_score = graph1_df[['episode_no', 'Score','rolling100_Score']].plot(x='episode_no', rot=0, 
                              color=[color_dict_graph1.get(x, '#9D9899') for x in['Score', 'rolling100_Score']],zorder=3,
                              kind='line',style ='-',fontsize=14,figsize=(8,5),secondary_y=True,ax =ax_graph1_step)



    plt.hlines(y=200, xmin=500,xmax=len(graph1_df), color='#9D9899', linestyle='--',lw=2,zorder =-1)
    plt.vlines(x=training_model_info.loc[((training_model_info["model_name"] == "LeakyReLU_LunarLander_v2_base") & (training_model_info['reached_goal'] ==True)), ]["episode_no"],
               ymin=graph1_df.Score.min(),ymax=graph1_df.Score.max(), color='#9D9899', linestyle='--',lw=2,zorder =-1)


    ax_graph1_step.set_title('Deep-Q-Network training',fontsize=14)
    ax_graph1_step.set_xlabel('Episode number',fontsize=14)
    ax_graph1_step.set_ylabel('Steps per episode',fontsize=14)
    ax_graph1_score.set_ylabel('Score',fontsize=14)

    #ax_graph1_step.legend(loc = "upper left")
    plt.savefig('./model_data/test_data_check/P2_graph1.png', bbox_inches='tight')
    plt.show()



def graph_2():
    #graph2_200_df


    graph2_200_df = test_model_info_200[test_model_info_200["model_name"] == "LeakyReLU_LunarLander_v2_base"][['episode_no', 'Score', 'step_count']]
    graph2_200_df["Rolling_20_score"] = graph2_200_df["Score"].rolling(20).mean()

    color_dict_graph2 = {'Rolling_20_score': '#CB4335', 'Score': '#F2D7D5'}


    plt.figure()
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},figsize=(8,5))
    plt.rcParams.update({'font.size': 14})
    ax_graph2_score_200 = graph2_200_df[['episode_no', 'Score','Rolling_20_score']].plot(x='episode_no', rot=0, 
                              color=[color_dict_graph2.get(x, '#333333') for x in['Score','Rolling_20_score']],
                              kind='line',style ='-',fontsize=14, ax= a0)

    a1.violinplot(dataset = [graph2_200_df['step_count']])

    #,subplots=True, layout=(1,2))
    #ax_graph2_score_250 = graph2_250_df[['episode_no', 'Score']].plot(x='episode_no', rot=0, 
    #                          color="#D815CC",zorder=3,
    #                          kind='line',style ='-',fontsize=14,figsize=(8,6),ax =ax_graph2_score_200)



    ax_graph2_score_200.set_title('Deep-Q-Network testing',fontsize=14)
    ax_graph2_score_200.set_xlabel('Episode number',fontsize=14)
    ax_graph2_score_200.set_ylabel('Score',fontsize=14)
    a1.set_title('Step Count distribution',fontsize=14)
    #ax_graph2_score_200.set_ylabel('Score',fontsize=14)


    a1.yaxis.set_ticks_position("right")
    plt.subplots_adjust(wspace=0, hspace=0)


    #ax_graph1_step.legend(loc = "upper left")
    plt.savefig('P2_graph2.png', bbox_inches='tight')

    plt.show()

def repeat_1():
    plt.rcParams.update({'font.size': 14})
    f, (ax_repeat, a1_repeat) = plt.subplots(1, 2, figsize=(8,5),gridspec_kw={'width_ratios': [3, 1]})
    graph_repeat_exp1 = training_model_info[training_model_info.model_identifier == "repeat"]

    for name,group  in graph_repeat_exp1.groupby('identifier_param'):
            ax_repeat.plot(group['episode_no'] ,group['rolling_Score'], label = "seed"+name)
            #ax = group[['episode_no','rolling_Score']].plot(x='episode_no', rot=0, 
            #                  zorder=3,
            #                  kind='line',style ='-',fontsize=14,figsize=(8,6), ax = ax)
    ax_repeat.legend(fontsize=14)
    ax_repeat.set_title('Model Repeatability Comparison',fontsize=14)
    ax_repeat.set_xlabel('Episode number',fontsize=14)
    ax_repeat.set_ylabel('Rolling Avg Score (100 episodes)',fontsize=14)
    #ax_repeat.yaxis.grid(True)



    ax_repeat.hlines(y=200, xmin=graph_repeat_exp1["episode_no"].min(),xmax=graph_repeat_exp1["episode_no"].max(), color='#9D9899', linestyle='--',lw=1,zorder =-1)

    a1_repeat.violinplot(dataset = [graph_repeat_exp1[graph_repeat_exp1['reached_goal'] ==True].episode_no])
    a1_repeat.set_title('#Eps. to reach 200',fontsize=14)
    a1_repeat.yaxis.set_ticks_position("right")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('P2_graph_repeat_ext.png', bbox_inches='tight')

    plt.show()


def compare_200_250():
    f, (a_repeat_score, a_repeat_step) = plt.subplots(1, 2, figsize=(4,5),gridspec_kw={'width_ratios': [1, 1]})

    graph_repeat_exp200 = test_model_info_200[test_model_info_200.model_identifier == "repeat"]
    graph_repeat_exp250 = test_model_info_250[test_model_info_250.model_identifier == "repeat"]

    a_repeat_score.violinplot(dataset = [graph_repeat_exp200.Score,graph_repeat_exp250.Score],widths=1)
    a_repeat_score.set_xticklabels([0,200,250])
    #a_repeat_score.hlines(y=200, xmin=199,xmax=300, color='#9D9899', linestyle='--',lw=1,zorder =-1)
    #a_repeat_score.yaxis.grid(True)
    a_repeat_step.yaxis.set_ticks_position("right")
    a_repeat_step.yaxis.set_label_position("right")

    a_repeat_step.violinplot(dataset = [graph_repeat_exp200.step_count,graph_repeat_exp250.step_count],widths=1)
    a_repeat_step.set_xticklabels([0,200,250])
    #a_repeat_step.yaxis.grid(True)
    a_repeat_score.set_xlabel('goals',fontsize=14)
    #with 10 differnt training itterations
    a_repeat_score.set_ylabel('Scores Distribution',fontsize=14)

    a_repeat_step.set_xlabel('goals',fontsize=14)
    a_repeat_step.set_ylabel('Steps per episode',fontsize=14)
    f.suptitle("Performance comparison\n of model with differnt training goals", fontsize=14)
    plt.savefig('P2_graph.png', bbox_inches='tight')
    #,\n over 10 different itterations of 100 episodes"

    a_repeat_score.hlines(y=200, xmin=1,xmax=2, color='#9D9899', linestyle='--',lw=1,zorder =-1)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig('P2_graph_repeat_ext_2.png', bbox_inches='tight')

    plt.show()


def repeat_2():
    repeat_test_grapher_df = test_model_info_250[test_model_info_250["model_identifier"] == "repeat"].set_index("episode_no")
    #repeat_test_grapher_df.groupby('identifier_param')["step_count"].plot(legend=True)

    f_rtg,(a_rtg_score,a_rtg_steps) = plt.subplots(1, 2, figsize=(8,5),gridspec_kw={'width_ratios': [1, 1]})
    plt.rcParams.update({'font.size': 14})


    a_rtg_score.violinplot(dataset = [group["Score"] for name, group in repeat_test_grapher_df.groupby('identifier_param')],widths=1)
    a_rtg_steps.violinplot(dataset = [group["step_count"] for name, group in repeat_test_grapher_df.groupby('identifier_param')],widths=1)

    a_rtg_steps.yaxis.set_ticks_position("right")
    a_rtg_steps.yaxis.set_label_position("right")

    a_rtg_score.set_xlabel('Model Seeds',fontsize=14)
    #with 10 differnt training itterations
    a_rtg_score.set_ylabel('Scores Distribution',fontsize=14)

    a_rtg_steps.set_xlabel('Model Seeds',fontsize=14)
    a_rtg_steps.set_ylabel('Steps per episode',fontsize=14)
    f_rtg.suptitle("Model performance comparisonwith differnt training exeperience", fontsize=14)

    plt.subplots_adjust(wspace=0, hspace=0)

    a_rtg_score.hlines(y=200, xmin=1,xmax=10, color='#9D9899', linestyle='--',lw=1,zorder =-1)
    plt.savefig('P2_graph_repeat_ext_3.png', bbox_inches='tight')


    plt.show()


def skip_test_1():


    plt.figure()

    plt.rcParams.update({'font.size': 10})

    Experiment_1_C_training_summary =   table_for_analysis[table_for_analysis["model_identifier"] == "C" ]
    f_exp1ct1,(a_exp1ct1_eps_step,a_exp1ct1_time) = plt.subplots(2,1, figsize=(5,5), gridspec_kw={'height_ratios': [4,1]}, sharex=True)

    #Experiment_1_C_training_summary.sort_values("identifier_param")


    Experiment_1_C_training_summary_episode = Experiment_1_C_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['episode_no'] ,logx=True,
                                                                                                                        style ='-',marker ="o",fontsize=14,
                                      markerfacecolor='yellow', legend=True, ax = a_exp1ct1_eps_step,
                                                                                                                        x_compat=True)
    Experiment_1_C_training_summary_step = Experiment_1_C_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['step_count_cumsum'],
                                                                                                secondary_y=True,ax =Experiment_1_C_training_summary_episode,
                                                                                                                    marker ="o",fontsize=14,
                                      markerfacecolor='red', legend=True
                                                                                                                    )

    Experiment_1_C_training_summary_episode_ticks =list(Experiment_1_C_training_summary.sort_values("identifier_param")["identifier_param"])
    Experiment_1_C_training_summary_episode.set_xticklabels(Experiment_1_C_training_summary_episode_ticks)
    Experiment_1_C_training_summary_episode.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(Experiment_1_C_training_summary_episode_ticks)
    Experiment_1_C_training_summary_episode.set_ylim(0,6000)
    Experiment_1_C_training_summary_episode.set_title('Performance of model at Skip training intervals\n',fontsize=14)
    Experiment_1_C_training_summary_episode.set_ylabel('#episode to reach 250 avg score\n for 100 consicutive runs, or end of training',fontsize=14)
    Experiment_1_C_training_summary_step.set_ylabel('Cumilative number of steps',fontsize=14)



    a_exp1ct1_time.set_xlabel('Skip interval (C)',fontsize=14)
    Experiment_1_C_training_summary_time = Experiment_1_C_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['exec_time_cumsum'],
                                                                                                secondary_y=True,
                                                                                                                    marker ="o",fontsize=12,
                                      markerfacecolor='red', legend=False, ax =a_exp1ct1_time
                                                                                                                    )

    #Experiment_1_C_training_summary_step.set_xlabel('Skip interval (C)',fontsize=14)
    a_exp1ct1_time.set_xlabel('Skip interval (C)',fontsize=14)
    Experiment_1_C_training_summary_time.set_ylabel('Cumilative\nlearning\ntime',fontsize=14)

    #ax_graph1_score.set_ylabel('Score',fontsize=14)
    #ax_graph1_step.legend(loc = "upper left")

    plt.figtext(.135, .5, "C= 50 and 100\n did not meet the goal\n requirements")
    plt.subplots_adjust(wspace=0, hspace=0.02)
    #plt.savefig('P2_graph1.png', bbox_inches='tight')



    plt.savefig('P2_experiment_1_C_1.png', bbox_inches='tight')
    plt.show()


def skip_test_2():
    Experiment_1_C_test_summary = test_model_info_250[test_model_info_250["model_identifier"] == "C"].set_index("episode_no")
    #repeat_test_grapher_df.groupby('identifier_param')["step_count"].plot(legend=True)
    Experiment_1_C_test_summary['identifier_param'] = Experiment_1_C_test_summary['identifier_param'].astype("int")

    Experiment_1_C_test_summary.sort_values('identifier_param',inplace= True)


    f_exp1ct,(a_exp1ct_steps,a_exp1ct_score) = plt.subplots(2, figsize=(5,5))
    plt.rcParams.update({'font.size': 14})
    #dataset = 

    label_names_a_exp1ct_score = list([name for name, group in Experiment_1_C_test_summary.groupby('identifier_param')])

    a_exp1ct_score.violinplot([group["Score"] for name, group in Experiment_1_C_test_summary.groupby('identifier_param')],
                              range(len(label_names_a_exp1ct_score)),widths=1, vert=True)


    a_exp1ct_score.set_xticks(range(len(label_names_a_exp1ct_score)))

    a_exp1ct_score.set_xticklabels(label_names_a_exp1ct_score)



    a_exp1ct_steps.violinplot([group["step_count"] for name, group in Experiment_1_C_test_summary.groupby('identifier_param')],
                             range(len(label_names_a_exp1ct_score)),widths=1, vert=True)

    #a_exp1ct_steps.yaxis.set_ticks_position("right")
    #a_exp1ct_steps.yaxis.set_label_position("right")

    a_exp1ct_score.set_xlabel('Skip interval (C)',fontsize=14)
    #with 10 differnt training itterations
    a_exp1ct_score.set_ylabel('Scores Distribution',fontsize=14)

    a_exp1ct_steps.set_ylabel('Avg #Steps/eps\n Distribution',fontsize=14)
    a_exp1ct_steps.xaxis.set_visible(False)

    #a_exp1ct_steps.legend([name for name, group in Experiment_1_C_test_summary.groupby('identifier_param')])
    #a_exp1ct_score.legend([name for name, group in Experiment_1_C_test_summary.groupby('identifier_param')])


    f_exp1ct.suptitle("Performance comparison of model\n with differnt skip interval", fontsize=14)

    plt.subplots_adjust(wspace=0, hspace=0)

    a_exp1ct_score.hlines(y=200, xmin=0,xmax=len(label_names_a_exp1ct_score)-1, color='#9D9899', linestyle='--',lw=1,zorder =-1)
    plt.savefig('P2_experiment_1_C_2.png', bbox_inches='tight')


    plt.show()



def target_1():

    plt.figure()

    plt.rcParams.update({'font.size': 10})

    Experiment_1_N_training_summary =   table_for_analysis[table_for_analysis["model_identifier"] == "_N" ]
    f_exp2nt1,(a_exp2nt1_eps_step,a_exp2nt1_time) = plt.subplots(2,1, figsize=(5,5), gridspec_kw={'height_ratios': [4,1]}, sharex=True)

    #Experiment_1_C_training_summary.sort_values("identifier_param")


    Experiment_1_N_training_summary_episode = Experiment_1_N_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['episode_no'] ,logx=True,
                                                                                                                        style ='-',marker ="o",fontsize=14,
                                      markerfacecolor='yellow', legend=True, ax = a_exp2nt1_eps_step,
                                                                                                                        x_compat=True)
    Experiment_1_N_training_summary_step = Experiment_1_N_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['step_count_cumsum'],
                                                                                                secondary_y=True,ax =Experiment_1_N_training_summary_episode,
                                                                                                                    marker ="o",fontsize=14,
                                      markerfacecolor='red', legend=True
                                                                                                                    )

    Experiment_1_N_training_summary_episode_ticks =list(Experiment_1_N_training_summary.sort_values("identifier_param")["identifier_param"])
    Experiment_1_N_training_summary_episode.set_xticklabels(Experiment_1_N_training_summary_episode_ticks)
    Experiment_1_N_training_summary_episode.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(Experiment_1_N_training_summary_episode_ticks)
    #Experiment_1_C_training_summary_episode.set_ylim(0,6000)
    Experiment_1_N_training_summary_episode.set_title('Performance of model with\nTarget DQN training intervals\n',fontsize=14)
    Experiment_1_N_training_summary_episode.set_ylabel('#episode to reach 250 avg score\n for 100 consicutive runs, or end of training',fontsize=14)
    Experiment_1_N_training_summary_step.set_ylabel('Cumilative number of steps',fontsize=14)




    Experiment_1_N_training_summary_time = Experiment_1_N_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['exec_time_cumsum'],
                                                                                                secondary_y=True,
                                                                                                                    marker ="o",fontsize=12,
                                      markerfacecolor='red', legend=False, ax =a_exp2nt1_time
                                                                                                                    )


    #Experiment_1_N_training_summary_time.set_xlabel('Skip interval (C)',fontsize=14)
    Experiment_1_N_training_summary_time.set_ylabel('Cumilative\nlearning\ntime',fontsize=12)

    #ax_graph1_score.set_ylabel('Score',fontsize=14)
    #ax_graph1_step.legend(loc = "upper left")

    #plt.figtext(.135, .5, "C= 50 and 100\n did not meet the goal\n requirements")
    plt.subplots_adjust(wspace=0, hspace=0.02)
    a_exp2nt1_time.set_xlabel('Target Network update interval (N)',fontsize=12)



    #plt.ticklabel_format(style='sci', axis='x', scilimits=(1,1))
    #plt.savefig('P2_graph1.png', bbox_inches='tight')
    plt.savefig('P2_experiment_2_N_1.png', bbox_inches='tight')
    plt.show()


def target_2():
    Experiment_1_N_test_summary = test_model_info_250[test_model_info_250["model_identifier"] == "_N"].set_index("episode_no")
    #repeat_test_grapher_df.groupby('identifier_param')["step_count"].plot(legend=True)
    Experiment_1_N_test_summary['identifier_param'] = Experiment_1_N_test_summary['identifier_param'].astype("int")

    Experiment_1_N_test_summary.sort_values('identifier_param',inplace= True)

    plt.rcParams.update({'font.size': 14})
    f_exp2nt2,(a_exp2nt2_steps,a_exp2nt2_score) = plt.subplots(2, figsize=(7,5), sharex=True)

    #dataset = 

    label_names_a_exp2nt2_score = list([name for name, group in Experiment_1_N_test_summary.groupby('identifier_param')])

    a_exp2nt2_score.violinplot([group["Score"] for name, group in Experiment_1_N_test_summary.groupby('identifier_param')],
                              range(len(label_names_a_exp2nt2_score)),widths=1, vert=True)


    a_exp2nt2_score.set_xticks(range(len(label_names_a_exp2nt2_score)))

    a_exp2nt2_score.set_xticklabels(label_names_a_exp2nt2_score)



    a_exp2nt2_steps.violinplot([group["step_count"] for name, group in Experiment_1_N_test_summary.groupby('identifier_param')],
                             range(len(label_names_a_exp2nt2_score)),widths=1, vert=True)

    #a_exp2nt2_steps.yaxis.set_ticks_position("right")
    #a_exp2nt2_steps.yaxis.set_label_position("right")

    a_exp2nt2_score.set_xlabel('Target Network update interval (N)',fontsize=14)
    #with 10 differnt training itterations
    a_exp2nt2_score.set_ylabel('Scores Distribution',fontsize=14)

    a_exp2nt2_steps.set_ylabel('Avg #Steps/eps\n Distribution',fontsize=14)
    a_exp2nt2_steps.xaxis.set_visible(False)

    #a_exp2nt2_steps.legend([name for name, group in Experiment_1_N_test_summary.groupby('identifier_param')])
    #a_exp2nt2_score.legend([name for name, group in Experiment_1_N_test_summary.groupby('identifier_param')])


    f_exp2nt2.suptitle("Performance comparison of model\n with differnt Target Network update interval", fontsize=14)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.hlines(y=200, xmin=0,xmax=len(label_names_a_exp2nt2_score)-1, color='#9D9899', linestyle='--',lw=1,zorder =-1)


    plt.savefig('P2_experiment_2_N_2.png', bbox_inches='tight')


    plt.show()

def batch_test_1():

    plt.figure()

    plt.rcParams.update({'font.size': 10})

    Experiment_3_batch_training_summary =   table_for_analysis[table_for_analysis["model_identifier"] == "batch_size" ]
    f_exp3bt1,(a_exp3bt1_eps_step,a_exp3bt1_time) = plt.subplots(2,1, figsize=(5,5), gridspec_kw={'height_ratios': [4,1]}, sharex=True)

    #Experiment_1_C_training_summary.sort_values("identifier_param")


    Experiment_3_batch_training_summary_episode = Experiment_3_batch_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['episode_no'] ,logy=False,
                                                                                                                        style ='-',marker ="o",fontsize=14,
                                      markerfacecolor='yellow', legend=True, ax = a_exp3bt1_eps_step,
                                                                                                                        x_compat=True)
    Experiment_3_batch_training_summary_step = Experiment_3_batch_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['step_count_cumsum'],
                                                                                                secondary_y=True,ax =Experiment_3_batch_training_summary_episode,
                                                                                                                    marker ="o",fontsize=14,
                                      markerfacecolor='red', legend=True
                                                                                                                    )

    Experiment_3_batch_training_summary_episode_ticks =list(Experiment_3_batch_training_summary.sort_values("identifier_param")["identifier_param"])
    #Experiment_3_batch_training_summary_episode.set_xticks(range(len(Experiment_3_batch_training_summary_episode_ticks)))
    #Experiment_3_batch_training_summary_episode.set_xticklabels(Experiment_3_batch_training_summary_episode_ticks)

    plt.xticks(Experiment_3_batch_training_summary_episode_ticks)
    Experiment_3_batch_training_summary_episode.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    Experiment_3_batch_training_summary_episode.set_ylim(0,3000)
    Experiment_3_batch_training_summary_step.set_ylim(0,1500000)
    Experiment_3_batch_training_summary_episode.set_title('Performance of model at different batch size',fontsize=14)
    Experiment_3_batch_training_summary_episode.set_ylabel('#episode to reach 250 avg score\n for 100 consicutive runs, or end of training',fontsize=12)
    Experiment_3_batch_training_summary_step.set_ylabel('Cumilative number of steps',fontsize=12)




    Experiment_3_batch_training_summary_time = Experiment_3_batch_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['exec_time_cumsum'],
                                                                                                secondary_y=True,
                                                                                                                    marker ="o",fontsize=12,
                                      markerfacecolor='red', legend=False, ax =a_exp3bt1_time
                                                                                                                    )


    #Experiment_3_batch_training_summary_time.set_xlabel('Skip interval (C)',fontsize=14)
    Experiment_3_batch_training_summary_time.set_ylabel('Cumilative\nlearning\ntime (s)',fontsize=12)

    #ax_graph1_score.set_ylabel('Score',fontsize=14)
    #ax_graph1_step.legend(loc = "upper left")

    plt.figtext(.24, .6, "batch_size= 8 did not meet the goal\n requirements,under 6000 episodes")
    plt.subplots_adjust(wspace=0, hspace=0.02)
    a_exp3bt1_time.set_xlabel('Batch Size for learning',fontsize=14)
    Experiment_3_batch_training_summary_time.set_ylim(0,1500)
    #plt.savefig('P2_graph1.png', bbox_inches='tight')

    plt.savefig('P2_experiment_3_batch_size_1.png', bbox_inches='tight')
    plt.show()

def batch_test_2():
    plt.rcParams.update({'font.size': 14})
    f, ax_batch_size = plt.subplots(1, figsize=(7,4))
    graph_batch_size_exp1 = training_model_info[training_model_info.model_identifier == "batch_size"]

    for name,group  in graph_batch_size_exp1.groupby('identifier_param'):
            ax_batch_size.plot(group['episode_no'] ,group['rolling_Score'], label = "batch_size"+str(name))
            #ax = group[['episode_no','rolling_Score']].plot(x='episode_no', rot=0, 
            #                  zorder=3,
            #                  kind='line',style ='-',fontsize=14,figsize=(8,6), ax = ax)
    ax_batch_size.legend(fontsize=10)
    ax_batch_size.set_title('training progressions with different Batch_size',fontsize=14)
    ax_batch_size.set_xlabel('Episode number',fontsize=14)
    ax_batch_size.set_ylabel('Rolling Avg Score (100 episodes)',fontsize=14)
    #ax_batch_size.yaxis.grid(True)



    ax_batch_size.hlines(y=200, xmin=graph_batch_size_exp1["episode_no"].min(),xmax=graph_batch_size_exp1["episode_no"].max(), color='#9D9899', linestyle='--',lw=1,zorder =-1)

    #a1_batch_size.violinplot(dataset = [graph_batch_size_exp1[graph_batch_size_exp1['reached_goal'] ==True].episode_no])
    #a1_batch_size.set_title('#Eps. to reach 200',fontsize=14)
    #a1_batch_size.yaxis.set_ticks_position("right")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('P2_experiment_3_batch_size_1_1.png', bbox_inches='tight')

    plt.show()

    
    
    
def batch_test_3():
    Experiment_3_batch_size_test_summary = test_model_info_250[test_model_info_250["model_identifier"] == "batch_size"].set_index("episode_no")
    #repeat_test_grapher_df.groupby('identifier_param')["step_count"].plot(legend=True)
    Experiment_3_batch_size_test_summary['identifier_param'] = Experiment_3_batch_size_test_summary['identifier_param'].astype("int")

    Experiment_3_batch_size_test_summary.sort_values('identifier_param',inplace= True)

    plt.rcParams.update({'font.size': 14})
    f_exp3bt2,(a_exp3bt2_steps,a_exp3bt2_score) = plt.subplots(2, figsize=(5,5), sharex=True)

    #dataset = 

    label_names_a_exp3bt2_score = list([name for name, group in Experiment_3_batch_size_test_summary.groupby('identifier_param')])

    a_exp3bt2_score.violinplot([group["Score"] for name, group in Experiment_3_batch_size_test_summary.groupby('identifier_param')],
                              range(len(label_names_a_exp3bt2_score)),widths=1, vert=True)


    a_exp3bt2_score.set_xticks(range(len(label_names_a_exp3bt2_score)))

    a_exp3bt2_score.set_xticklabels(label_names_a_exp3bt2_score)



    a_exp3bt2_steps.violinplot([group["step_count"] for name, group in Experiment_3_batch_size_test_summary.groupby('identifier_param')],
                             range(len(label_names_a_exp3bt2_score)),widths=1, vert=True)

    #a_exp3bt2_steps.yaxis.set_ticks_position("right")
    #a_exp3bt2_steps.yaxis.set_label_position("right")

    a_exp3bt2_score.set_xlabel('Batch Size for learning',fontsize=14)
    #with 10 differnt training itterations
    a_exp3bt2_score.set_ylabel('Scores Distribution',fontsize=14)

    a_exp3bt2_steps.set_ylabel('Avg #Steps/eps\n Distribution',fontsize=14)
    a_exp3bt2_steps.xaxis.set_visible(False)

    #a_exp3bt2_steps.legend([name for name, group in Experiment_3_batch_size_test_summary.groupby('identifier_param')])
    #a_exp3bt2_score.legend([name for name, group in Experiment_3_batch_size_test_summary.groupby('identifier_param')])


    f_exp3bt2.suptitle("Performance comparison of model\n with differnt batch sizes for learning", fontsize=14)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.hlines(y=200, xmin=0,xmax=len(label_names_a_exp3bt2_score)-1, color='#9D9899', linestyle='--',lw=1,zorder =-1)


    plt.savefig('P2_experiment_3_batch_size_2.png', bbox_inches='tight')


    plt.show()


def replay_memory_test_1():

    plt.figure()

    plt.rcParams.update({'font.size': 10})

    Experiment_2_replay_memory_size_training_summary =   table_for_analysis[table_for_analysis["model_identifier"] == "replay_memory_size" ]
    f_exp4rmt1,(a_exp4rmt1_eps_step,a_exp4rmt1_time) = plt.subplots(2,1, figsize=(5,5), gridspec_kw={'height_ratios': [4,1]}, sharex=True)

    #Experiment_2_C_training_summary.sort_values("identifier_param")


    Experiment_2_replay_memory_size_training_summary_episode = Experiment_2_replay_memory_size_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['episode_no'] ,logx=True,
                                                                                                                        style ='-',marker ="o",fontsize=14,
                                      markerfacecolor='yellow', legend=True, ax = a_exp4rmt1_eps_step,
                                                                                                                        x_compat=True)
    Experiment_2_replay_memory_size_training_summary_step = Experiment_2_replay_memory_size_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['step_count_cumsum'],
                                                                                                secondary_y=True,ax =Experiment_2_replay_memory_size_training_summary_episode,
                                                                                                                    marker ="o",fontsize=14,
                                      markerfacecolor='red', legend=True
                                                                                                                    )

    Experiment_2_replay_memory_size_training_summary_episode_ticks =list(Experiment_2_replay_memory_size_training_summary.sort_values("identifier_param")["identifier_param"])
    Experiment_2_replay_memory_size_training_summary_episode.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(Experiment_2_replay_memory_size_training_summary_episode_ticks)
    #Experiment_2_replay_memory_size_training_summary_episode.set_xticklabels(Experiment_2_replay_memory_size_training_summary_episode_ticks,rotation=45)

    #Experiment_2_C_training_summary_episode.set_ylim(0,6000)
    Experiment_2_replay_memory_size_training_summary_episode.set_title('Performance of model at Skip training intervals\n',fontsize=14)
    Experiment_2_replay_memory_size_training_summary_episode.set_ylabel('#episode to reach 250 avg score\n for 100 consicutive runs, or end of training',fontsize=14)
    Experiment_2_replay_memory_size_training_summary_step.set_ylabel('Cumilative number of steps',fontsize=14)




    Experiment_2_replay_memory_size_training_summary_time = Experiment_2_replay_memory_size_training_summary.sort_values("identifier_param").plot.line(x='identifier_param',y =['exec_time_cumsum'],
                                                                                                secondary_y=True,
                                                                                                                    marker ="o",fontsize=12,
                                      markerfacecolor='red', legend=False, ax =a_exp4rmt1_time
                                                                                                                    )


    #Experiment_2_replay_memory_size_training_summary_time.set_xlabel('Skip interval (C)',fontsize=14)
    Experiment_2_replay_memory_size_training_summary_time.set_ylabel('Cumilative\nlearning\ntime',fontsize=12)

    #ax_graph1_score.set_ylabel('Score',fontsize=14)
    #ax_graph1_step.legend(loc = "upper left")

    plt.figtext(.3, .6, "Large as well as small\nreplay memory adversely effects\nmodel training time")
    plt.subplots_adjust(wspace=0, hspace=0.02)
    a_exp4rmt1_time.set_xlabel('Replay memory size',fontsize=14)
    plt.xticks(rotation=40)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('P2_experiment_3_replay_memory_1.png', bbox_inches='tight')

    plt.show()


def replay_memory_test_2():
    plt.rcParams.update({'font.size': 14})
    f, ax_replay_memory_size = plt.subplots(1, figsize=(6,5))
    graph_replay_memory_size_exp1 = training_model_info[training_model_info.model_identifier == "replay_memory_size"]

    for name,group  in graph_replay_memory_size_exp1.groupby('identifier_param'):
            ax_replay_memory_size.plot(group['episode_no'] ,group['rolling_Score'], label = "replay mem.size:"+str(int(name) ))
            #ax = group[['episode_no','rolling_Score']].plot(x='episode_no', rot=0, 
            #                  zorder=3,
            #                  kind='line',style ='-',fontsize=14,figsize=(8,6), ax = ax)
    ax_replay_memory_size.legend(fontsize=10)
    ax_replay_memory_size.set_title('Model tranning with differnt Replay memory size',fontsize=14)
    ax_replay_memory_size.set_xlabel('Episode number',fontsize=14)
    ax_replay_memory_size.set_ylabel('Rolling Avg Score (100 episodes)',fontsize=14)
    #ax_replay_memory_size.yaxis.grid(True)



    ax_replay_memory_size.hlines(y=200, xmin=graph_replay_memory_size_exp1["episode_no"].min(),xmax=graph_replay_memory_size_exp1["episode_no"].max(), color='#9D9899', linestyle='--',lw=1,zorder =-1)

    #a1_replay_memory_size.violinplot(dataset = [graph_replay_memory_size_exp1[graph_replay_memory_size_exp1['reached_goal'] ==True].episode_no])
    #a1_replay_memory_size.set_title('#Eps. to reach 200',fontsize=14)
    #a1_replay_memory_size.yaxis.set_ticks_position("right")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('P2_experiment_3_replay_memory_1_1.png', bbox_inches='tight')

    plt.show()


    
    
def replay_memory_3():
    Experiment_4_replay_memory_size_test_summary = test_model_info_250[test_model_info_250["model_identifier"] == "replay_memory_size"].set_index("episode_no")
    #repeat_test_grapher_df.groupby('identifier_param')["step_count"].plot(legend=True)
    Experiment_4_replay_memory_size_test_summary['identifier_param'] = Experiment_4_replay_memory_size_test_summary['identifier_param'].astype("int")

    Experiment_4_replay_memory_size_test_summary.sort_values('identifier_param',inplace= True)

    plt.rcParams.update({'font.size': 14})
    f_exp4rmt2,(a_exp4rmt2_steps,a_exp4rmt2_score) = plt.subplots(2, figsize=(5,5), sharex=True)

    #dataset = 

    label_names_a_exp4rmt2_score = list([int(name)/100000 for name, group in Experiment_4_replay_memory_size_test_summary.groupby('identifier_param')])

    a_exp4rmt2_score.violinplot([group["Score"] for name, group in Experiment_4_replay_memory_size_test_summary.groupby('identifier_param')],
                              range(len(label_names_a_exp4rmt2_score)),widths=1, vert=True)


    a_exp4rmt2_score.set_xticks(range(len(label_names_a_exp4rmt2_score)))

    a_exp4rmt2_score.set_xticklabels(label_names_a_exp4rmt2_score)



    a_exp4rmt2_steps.violinplot([group["step_count"] for name, group in Experiment_4_replay_memory_size_test_summary.groupby('identifier_param')],
                             range(len(label_names_a_exp4rmt2_score)),widths=1, vert=True)

    #a_exp4rmt2_steps.yaxis.set_ticks_position("right")
    #a_exp4rmt2_steps.yaxis.set_label_position("right")

    a_exp4rmt2_score.set_xlabel('Replay memory size [1e5]',fontsize=14)
    #with 10 differnt training itterations
    a_exp4rmt2_score.set_ylabel('Scores Distribution',fontsize=14)

    a_exp4rmt2_steps.set_ylabel('Avg #Steps/eps\n Distribution',fontsize=14)
    a_exp4rmt2_steps.xaxis.set_visible(False)

    #a_exp4rmt2_steps.legend([name for name, group in Experiment_4_replay_memory_size_test_summary.groupby('identifier_param')])
    #a_exp4rmt2_score.legend([name for name, group in Experiment_4_replay_memory_size_test_summary.groupby('identifier_param')])


    f_exp4rmt2.suptitle("Performance comparison of model\n with differnt Replay memory size", fontsize=14)

    plt.subplots_adjust(wspace=0, hspace=0)


    a_exp4rmt2_score.hlines(y=200, xmin=0,xmax=len(label_names_a_exp4rmt2_score)-1, color='#9D9899', linestyle='--',lw=1,zorder =-1)

    plt.savefig('P2_experiment_3_replay_memory_2.png', bbox_inches='tight')

    plt.show()
    
    

    
    
def run_all():
    base_test()
    repeat_test()
    gamma_test()
    ep_decay()
    alpha_test()
    target_network()
    skip_test()
    neurons()
    batch_size()
    replay_memory()
    
    training_model_info, test_model_info_200,test_model_info_250 = get_data_from_csv(address_modifier=address_modifier)
    training_model_info,test_model_info_200,test_model_info_250,table_for_analysis_250,table_for_analysis_200,table_for_analysis,table_for_analysis_goal = data_handler()
    
    Auto_gif_creator( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*")
    #  Auto_test_all_models( env = gym.make('LunarLander-v2'), base_address = "./model_data/", address_modifier = "test_data_check/*")

    graph_1()
    graph_2()
    repeat_1()
    compare_200_250()
    repeat_2()
    skip_test_1()
    skip_test_2()
    target_1()
    target_2()
    batch_test_1()
    batch_test_2()
    batch_test_3()
    replay_memory_test_1()
    replay_memory_test_2()
    replay_memory_3()    
    
    
    
    
###############################################################################
#Running things

 

    