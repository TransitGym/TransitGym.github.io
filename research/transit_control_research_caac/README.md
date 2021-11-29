# Reducing Bus Bunching with Asynchronous Multi-Agent Reinforcement Learning

 
This is a Pytorch implementation of [Reducing Bus Bunching with Asynchronous Multi-Agent Reinforcement Learning](https://www.ijcai.org/proceedings/2021/0060.pdf)
## Requirements
- numpy>=1.19.2
- matplotlib>=3.3.3
- torch>=1.7.0
- pandas>=1.2.0

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Code Structure
- main.py :_Starting point of this program_
- model
    * ACCF.py :_Implementation of the proposed model_
    * DDPG.py :_Implementation of DDPG_
    * MADDPG.py :_Implementation of MADDPG_
    * Group_MemoryC.py:_Implementation of experience replay procedure_
    * layers.py: _Implementation of graph attention layer_
- sim
    * Bus.py :_Definition of bus related attributes and methods_
    * Busstop.py :_Definition of bus stop related attributes and methods_
    * Passenger.py :_Definition of passenger related attributes and methods_
    * Route.py :_Definition of bus route related attributes and methods_
    * Sim_Engine.py :_Basic logic of the simulation_
    * util.py :_Some assistant methods_
- result:Store results
    
## Data Introduction
### The data are not publicly available
The transit data files are available at `sim/data/` folder. For privacy reason, we have replaced the orignal real world data. The files to establish the simulation includes:
- demand.csv      : _Trip information of each passenger_ 
- stop_times.csv  : _Bus services information_
- stops.csv       : _Stop information_
- trips.csv       : _Route information_
 

## Model Training

Here are commands for training the reinforcement learning models. 

```bash
# Train the model
./train.sh


