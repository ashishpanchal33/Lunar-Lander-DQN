#please refer bellow for definitions

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
    
    
    
    
   
