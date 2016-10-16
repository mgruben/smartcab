import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# import packages for statistics and data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        '''
        sets self.env = env, state = None, next_waypoint = None,
        and a default color
        '''
        super(LearningAgent, self).__init__(env)
        
        # override color
        self.color = 'red'
        
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        
        # TODO: Initialize any additional variables here
        self.actions = (None, 'forward', 'left', 'right')
        self.state = None
        
        # Initialize variables for statistics tracking
        self.N = 100
        self.success = np.zeros(self.N)
        self.invalid = np.zeros(self.N)
        self.wander = np.zeros(self.N)
        self.trial = 0
        self.trips_failed = 0
        
        # Variables related to Q-learning
        self.Qtable = {} # The Q-Table
        self.weights = [0, 0, 0, 0] # stores preferences for current act
        self.Qiterations = {} # used to reduce alpha
        self.alpha = 1 # initial learning rate
        self.gamma = 0.03 # the discount factor
        self.optimism = 5 # the Q-Value to assign new states
        '''
        At a gamma of 1, the car remains stationary always.
        At a gamma of 0.9, the car very quickly favors looping.
        Even at a gamma of 0.33, the car begins to enter a looping
        behavior within 5 or 6 total trials
        At a gamma of 0.1, the tendency to loop is diminished, but
        looping occurs eventually after the ~12th trial.
        At a gamma of 0.03, it appears that the looping behaviour does
        not reappear, even up to 100 trials.
        '''

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if
        # required

    def get_state(self):
        '''
        Returns the state vector of the smartcab, according to
        (1) the next waypoint,
        (2) the color of the light,
        (3) the heading of traffic approaching from the left, and
        (4) the heading of oncoming traffic.
        '''
        
        # This is the only way the smartcab has an idea of which
        # direction it _should_ be taking.
        #
        # Otherwise, until the car reached the destination, it could
        # only receive negative feedback, such as when it collides with
        # another car, or when it disobeys a traffic signal.
        # 
        # Very quickly, then, this would lead the car to remain
        # stationary (and wait for the destination to come to it?).
        self.next_waypoint = self.planner.next_waypoint()
        
        # Gather inputs
        # from route planner, also displayed by simulator
        # Light color, and the heading of traffic, if any, from
        # (1) oncoming, (2) right, and (3) left
        inputs = self.env.sense(self)

        # return state with the suggested action along with
        # raw values from inputs, ordered to clockwise orientation
        return (self.next_waypoint, inputs['light'], 
            inputs['left'], inputs['oncoming'])
    
    def update(self, t):
        '''
        Takes the next "best" action as defined by the Q-Learning
        algorithm.
        
        First, the smartcab determines its state (that is, which action
        the route planner suggests next, the current light color, where
        the nearby cars are, and in which direction are those cars
        travelling).
        
        Next, the smartcab determines the cost or benefit of the
        potential actions it could take.  These potential actions,
        as defined in the constructor, are to remain stationary, or to
        go forward, left, or right.  For more information about the
        costs and benefits, consult environment.Environment.act().
        
        The smartcab then takes the action with the highest benefit.
        If multiple actions tie for the highest benefit, one of the
        tieing actions is chosen at random.
        
        The smartcab then peeks ahead to the highest possible benefit
        which can be obtained from the next action it would take,
        and uses that value to update its Q-Learning model.
        
        More precisely, the Q-Learning model is a table of state-action
        pairs, combined with preferences.  The adjustement of those
        preferences over time is the result of the Q-Learning algorithm.
        
        After a sufficiently-long training period (where the smartcab
        has explored the state-action pair space), it is the hope that
        the smartcab will "know" the correct action to take in order to
        get to its destination along the best route, without taking
        invalid (harmful, dangerous, illegal) actions.
        '''
        # Begin by initializing the smartcab's state vector
        self.state = self.get_state()
        
        # Get the current deadline
        deadline = self.env.get_deadline(self)
        
        # Select action according to your policy
        # If a state-action pair hasn't been considered yet, award
        # it a high initial Q value, to favor exploring new options
        #
        # While "exploring new options" is lamentable for a smartcab
        # with live passengers, it is also perhaps the only way to
        # traverse the reward-space during training.
        for i, action in enumerate(self.actions):
            self.weights[i] = \
                self.Qtable.setdefault((self.state, action), self.optimism)

        # The next action is the highest-Q-value action
        # if tie, the next action is randomly chosen from among ties
        current_max = max(self.weights)
        action_ties = [i for i, w in enumerate(self.weights) if w == current_max]
        action = self.actions[random.choice(action_ties)]
        
        # Keep track of how many times this action was chosen for
        # this state, and reduce alpha accordingly.
        #
        # If a state-action pair has never been seen before, then
        # the smartcab should learn as much as it can from this new
        # scenario.
        #
        # If a state-action pair has already occurred frequently,
        # then the smartcab should retain comparatively more of the
        # knowledge it already possesses about that state-action pair,
        # and should not "learn" as much from encountering the same
        # scenario again.
        iterations = self.Qiterations.setdefault((self.state, action),
            1)
        self.Qiterations[(self.state, action)] += 1
        self.alpha = 1.0 / iterations
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # Track statistics
        
        # Generally, we want to consider the trial over if the smartcab
        # reaches its destination.
        #
        # Generally, when the smartcab reaches its destination, it
        # receives a reward of 12.
        # This is generally because the smartcab (1) took a valid action
        # which agreed with the next waypoint (reward = 2), _and_
        # because it reached the destination (reward += 10).
        # 
        # However, the Route Planner is **not** perfect, and will
        # very rarely suggest a path to the smartcab which is not the
        # correct next move to get closer to the destination.
        # This is probably due to the grid being continuous, rather than
        # bounded and purely Euclidean, space (e.g. can travel left by
        # going far enough right).
        # 
        # For example:
        '''
        Simulator.run(): Trial 757
        Environment.reset(): Trial set up with start = (6, 6), destination = (6, 1), deadline = 25
        RoutePlanner.route_to(): destination = (6, 1)
        Environment.act(): Primary agent has reached destination!
        LearningAgent.update(): deadline = 25, state = ('right', 'green', 'right', None),  action = left, reward = 9.5 trial: = 758
        
        Simulator.run(): Trial 30
        Environment.reset(): Trial set up with start = (4, 6), destination = (3, 1), deadline = 30
        RoutePlanner.route_to(): destination = (3, 1)
        LearningAgent.update(): deadline = 30, state = ('left', 'red', None, None),  action = None, reward = 0.0 trial: = 30
        LearningAgent.update(): deadline = 29, state = ('left', 'red', None, None),  action = None, reward = 0.0 trial: = 30
        LearningAgent.update(): deadline = 28, state = ('left', 'red', None, None),  action = None, reward = 0.0 trial: = 30
        LearningAgent.update(): deadline = 27, state = ('left', 'red', None, None),  action = None, reward = 0.0 trial: = 30
        LearningAgent.update(): deadline = 26, state = ('left', 'green', None, None),  action = left, reward = 2.0 trial: = 30
        Environment.act(): Primary agent has reached destination!
        LearningAgent.update(): deadline = 25, state = ('right', 'green', 'right', None),  action = left, reward = 9.5 trial: = 31

        Simulator.run(): Trial 77
        Environment.reset(): Trial set up with start = (8, 2), destination = (3, 6), deadline = 45
        RoutePlanner.route_to(): destination = (3, 6)
        LearningAgent.update(): deadline = 45, state = ('right', 'red', None, None),  action = right, reward = 2.0 trial: = 77
        LearningAgent.update(): deadline = 44, state = ('forward', 'green', None, None),  action = forward, reward = 2.0 trial: = 77
        LearningAgent.update(): deadline = 43, state = ('forward', 'green', None, None),  action = forward, reward = 2.0 trial: = 77
        LearningAgent.update(): deadline = 42, state = ('forward', 'green', None, None),  action = forward, reward = 2.0 trial: = 77
        LearningAgent.update(): deadline = 41, state = ('forward', 'red', None, None),  action = None, reward = 0.0 trial: = 77
        LearningAgent.update(): deadline = 40, state = ('forward', 'red', None, None),  action = None, reward = 0.0 trial: = 77
        LearningAgent.update(): deadline = 39, state = ('forward', 'green', None, None),  action = forward, reward = 2.0 trial: = 77
        LearningAgent.update(): deadline = 38, state = ('left', 'red', None, 'forward'),  action = right, reward = -0.5 trial: = 77
        LearningAgent.update(): deadline = 37, state = ('right', 'red', None, 'left'),  action = forward, reward = -1.0 trial: = 77
        Environment.act(): Primary agent has reached destination!
        LearningAgent.update(): deadline = 36, state = ('right', 'green', None, 'left'),  action = forward, reward = 9.5 trial: = 78

        '''
        # In those certainstances, if the smartcab's next move is
        # (a) against the Route Planner (e.g. reward = -0.5), but
        # (b) gets the smartcab to the destination (reward += 10),
        # the smartcab will receive a reward of 9.5!!!
        #
        # Accordingly, we need to check whether the reward the smartcab
        # received on this iteration is greater than 9, because the
        # _only_ way that will occur is if the smartcab has, somehow,
        # reached the destination.
        
        if reward > 10: # destination reached by on-waypoint action
            self.success[self.trial] = 1
            self.trial += 1
        elif reward > 9: # destination reached by off-waypoint action
            self.success[self.trial] = 1
            self.trial += 1
            print "Negative reward: state = {}, action = {}, reward = {}".format(self.state, action, reward)
            self.wander[self.trial] += 1
        elif deadline == 0:
            self.trial += 1
            self.trips_failed += 1
        elif reward == -1:
            print "Negative reward: state = {}, action = {}, reward = {}".format(self.state, action, reward)
            self.invalid[self.trial] += 1
        elif reward == -0.5:
            print "Negative reward: state = {}, action = {}, reward = {}".format(self.state, action, reward)
            self.wander[self.trial] += 1
        
        
        # Learn policy based on state, action, reward
        
        # Peek ahead to store the next state to s' (state_prime)
        # 
        # It is important _not_ to set the state here, or else
        # we break the implementation of the Q-learning algorithm's
        # value updating
        state_prime = self.get_state()

        # Determine the utility of the new state.
        # 
        # While previously we were concerned with which action to take,
        # here we are concerned with the greatest possible Q-value from
        # any of the next actions we _could_ take.
        #
        # Here again, if we have not previously encountered a particular
        # state-action pair, we award a high initial Q value to favor
        # exploring new options.
        for i, action_prime in enumerate(self.actions):
            self.weights[i] = self.Qtable.setdefault((state_prime,
                action_prime), self.optimism)
        self.maxQ_new = max(self.weights)
        
        # Update Q for the current state with the just-calculated
        # utility for the next state
        # 
        # This is the equation from the "Estimating Q from Transitions"
        # Udacity video
        self.Qtable[(self.state, action)] = \
            (1.0 - self.alpha) * self.Qtable[(self.state, action)] + \
            self.alpha * (reward + self.gamma * self.maxQ_new)
        

        # print "LearningAgent.update(): " + \
        # "deadline = {}, state = {}, ".format(deadline, self.state) + \
        # " action = {}, reward = {}".format(action, reward)  # [debug]

def scatter(a, t):
    plt.plot(a, "o")
    plt.title(t)
    plt.show()

def nanCatOne(a):
    return [float('nan') if x == 1 else x for x in a]

def nanCatZero(a):
    return [float('nan') if x == 0 else x for x in a]
    
def allScatter(a, N):
    print(a.success)
    success = nanCatOne(a.success)
    print(success)
    m = 4
    l = len(success)
    w = 5
    plt.plot(success, "v", label="Trip Failed", color="green", markersize=2*m)
    
    plt.plot(pd.rolling_mean(a.invalid, window=w), label="Rolling invalid mean", color = "blue")
    plt.plot(pd.rolling_mean(a.wander, window=w), label="Rolling off-waypoint mean", color = "red")
    
    sum_invalid = int(sum(a.invalid))
    sum_wander = int(sum(a.wander))
    
    invalid = nanCatZero(a.invalid)
    wander = nanCatZero(a.wander)
    plt.plot(invalid, "|", label="Invalid actions", color="blue", markersize=2*m)
    plt.plot(wander, "|", label="Off-waypoint actions", color="red", markersize=2*m)
    
    t = str(N) + " trials, gamma = " + str(a.gamma) + "\n" + str(a.trips_failed) + " trips failed, " + str(sum_invalid) + " invalid actions, " + str(sum_wander) + " off-waypoint actions"
    plt.title(t)
    plt.ylabel("Number of Occurrences")
    plt.xlabel("Trial Number")
    
    plt.legend()
    
    plt.show()

def run():
    """Run the agent for a finite number of trials."""
    
    # create environment (also adds some dummy traffic)
    e = Environment()
    
    # create agent
    a = e.create_agent(LearningAgent)
    
    # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)
    # NOTE: You can set enforce_deadline=False while debugging to
    # allow longer trials
    
    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0, display=False)
    # NOTE: To speed up simulation, reduce update_delay and/or set
    # display=False
    N = 100
    sim.run(n_trials=N)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit
    # Ctrl+C on the command-line
    allScatter(a, N)

if __name__ == '__main__':
    run()
