## Teach a Quadcopter How to Fly Project Overview

The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue. Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors. 
But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. This is what you are going to do in this project!

### Evaluation

In this project, you will design your own reinforcement learning task and an agent to complete it. Note that getting a reinforcement learning agent to learn what you actually want it to learn can be hard, and very time consuming. For this project, we strongly encourage you to take the time to tweak your task and agent until your agent is able to demonstrate that it has learned your chosen task, but this is not necessary to complete the project. As long as you take the time to describe many attempts at specifying a reasonable reward function and a well-designed agent with well-informed hyperparameters, this is enough to pass the project.

