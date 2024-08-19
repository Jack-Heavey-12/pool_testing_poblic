# Using this version for time checks

import networkx as nx
import sys
from gurobipy import *
import math
import numpy as np
import time
import pandas as pd
import json
import random

import pickle


from collections import defaultdict
import itertools

from matplotlib import pyplot as plt

def enumerate(graph, n_p=3):
	#returns list of the possible subsets that can be chosen
	#NOTE: This list is O(n^{n_p}), be aware of memory constraints
	nodes = list(graph.nodes())
	#set_list = []
	set_list = list(itertools.combinations(nodes, n_p))
	print('Potential Pools Enumerated')
	return set_list

def enumerate_random(graph, n_p, num_sets=1000):
	#Returns a shorter list of possible subsets for when graphs are larger
	#TODO
	nodes = list(graph.nodes())
	set_list = []
	for _ in range(num_sets):
		set_list.append(tuple(random.sample(nodes, n_p)))
	return set_list


def cascade_construction(graph, N, p, source_count=1):
	# Returns the list of connected components to the source.
	# NOTE: NOT a nx.Graph() type
	cascade_list = []

	#Generates one graph per sample
	for i in range(N):
		TempGraph = nx.Graph()

		TempGraph.add_nodes_from(graph.nodes())

		#Adds an edge if it is randomly selected with probability p
		for j in graph.edges():
			r = np.random.random()

			if r <= p:
				TempGraph.add_edge(j[0], j[1])

		#NOTE: May need to correct this later, but I think random.choose is the correct, will return a list
		#		Hopefully shouldn't be any assignment issues either

		src = random.choices(list(graph.nodes()), k=source_count).copy()
		ccs = []
		for source in src:
			ccs.append(set(nx.node_connected_component(TempGraph, source)))
		#Quick thing to remove duplicates. Think it should work fine
		ccs_f = []
		total_set = set()
		for i in ccs:
			total_set.update(i)
		cascade_list.append(total_set)
		#cascade_list.append(ccs_f)


	return cascade_list



'''REQUIRED DATA TYPES FOR INPUTS:
	- graph: networkx.Graph()
	- set_list: list of tuples, possible available sets
	- cascades: A list of lists of connected components - they correspond directly to the nodes in the graph
	- B: int
	- overlapping: bool (defines whether overlapping sets are allowed or not, affects constraint 1)
'''
def LinearProgram(graph, set_list, cascades, B=3, overlapping=True):

	x = {} # defined in the paper
	y = {} # defined in the paper

	F_i = {} # Superset of F_vi, will define the sets here for each cascade. From here, only have to check for membership for the later constraint
	for i in range(len(cascades)):
		F_i[i] = [S for S in set_list if not any(x in cascades[i] for x in S)]

	N = len(cascades)

	m = Model('pool_testing')

	for S in set_list:
		x[S] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = f'x[{S}]')

	node_list = list(graph.nodes())
	for i in range(len(cascades)):
		for v in node_list:
			y[f'(node_{v}, casc_{i})'] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = f'y[({v}, {i})]')

	m.update()

	# Constraint 1
	#If we are doing overlapping sets
	'''if overlapping:
		for i in range(len(cascades)):
			for v in node_list:
				#print(cascades[i])
				val =  any([x if v in x else False for x in cascades[i]]) #Boolean for if the node is in ANY connected conponent from any source
				if not val:
					# The node is not in the connected component, any set with this node in it is valid:
					F_vi = [S for S in set_list if v in S]
					m.addConstr(y[f'({v}, {i})'] <= quicksum(x[S] for S in F_vi), name=f'C1_Node_{v}_cascade_{i}_overlapping')
	#If we have non-overlapping sets, the same thing but with equality in the constraint rather than an inequality	
	else:
		for i in range(len(cascades)):
			for v in node_list:
				val =  any([x if v in x else False for x in cascades[i]]) #Boolean for if the node is in ANY connected conponent from any source
				if not val:
					# The node is not in the connected component, any set with this node in it is valid:
					F_vi = [S for S in set_list if v in S]
					m.addConstr(y[f'({v}, {i})'] == quicksum(x[S] for S in F_vi), name=f'C1_Node_{v}_cascade_{i}')
	#Non-overlapping Constraint'''

	for i in range(len(cascades)):
			for v in node_list:
				val =  v in cascades[i] #Boolean for if the node is in the connected conponent from the source
				if not val:
					# The node is not in the connected component, any set with this node in it is valid:
					#F_vi = [S for S in set_list if v in S]
					F_vi = [S for S in F_i[i] if v in S] 
					m.addConstr(y[f'(node_{v}, casc_{i})'] <= quicksum(x[S] for S in F_vi), name=f'C1_Node_{v}_cascade_{i}')
	#END CONSTRAINT ONE

	m.update()
	

	# Constraint 2 - need results to be under the budget
	m.addConstr(quicksum(x[S] for S in set_list) <= B, name='C2_Budget Constraint')

	m.update()

	possible_cleared_nodes = []
	for i in range(len(cascades)):
		for v in node_list:
			val =  v in cascades[i]
			if not val:
				possible_cleared_nodes.append((v,i))

	m.setObjective(1/N * quicksum(y[f'(node_{v}, casc_{i})'] for (v,i) in possible_cleared_nodes), GRB.MAXIMIZE)
	m.setParam('OutputFlag', 1)
	m.update()
	m.optimize()
	print(f'Status Code: {m.Status}')

	print(f'Solution Count: {m.SolCount}')
	x_vals = {}
	y_vals = {}
	for i in x.keys():
		x_vals[i] = x[i].X
	for i in y.keys():
		y_vals[i] = y[i].X


	#RETURNS THE DICTIONARY WITH THE VARIABLES X (FOR ROUNDING LATER), DICTIONARY WITH VARIABLES Y, 
	#		AND THE OPTIMAL OBJECTIVE VALUE (UPPER BOUND ON ROUNDED ANSWER WITH NO VIOLATED BUDGET)
	variables = m.getVars()
	return x_vals, y_vals, m.objVal, variables


def rounding(x_dict):
	#Will return the rounded values in a dictionary, which correspond to specific subsets to choose
	#Doing this in a separate function in case the rounding function changes later

	x_prime_dict = {}
	for S in x.keys():
		limit = np.random.uniform(0,1)
		#print(type(x[S]))
		#print(dir(x[S]))
		#sys.exit()
		if limit <= x[S]:
			x_prime_dict[S] = 1
	#NOTE: 	This will not have *every* set in this dictionary, only the pools that we are going to end up choosing
	#		Figure this is easier than trying to check if every value is one or zero, we can jsut compare length for expectation, etc.

	#NOTE2:	The keys in x, and thus x_prime_dict, are the sets themselves. Think that's the best way to do it, shouldn't have any assignment issues?
	return x_prime_dict


def calculate_E_welfare(x_prime, cascade_list):
	#TODO
	num_cascades = len(cascade_list)
	running_clearances = 0
	for S in x_prime.keys():
		#x_prime.keys() is going to be a set, which means have to iterate through the set and then each cascade to see if it's cleared.
		set_S = set(S)
		for i in cascade_list:
			#if the intersection is empty, then add the running clearances (number of elements in the set)
			inter = set_S.intersection(i)
			if len(inter) == 0:	
				running_clearances += len(set_S)
	return running_clearances / num_cascades #The correct objective value I think?

	#Cycle through keys in x_prime, see which cascades they are cleared in

def read_graph(name):
	if name == 'test_graph':
		G = nx.read_edgelist('data/test_graph.txt')
	elif name == 'test_grid':
		G = nx.read_edgelist('data/test_grid.txt')
	elif name == 'lyon':
		df = pd.read_csv('data/hospital_contacts', sep='\t', header=None)
		df.columns = ['time', 'e1', 'e2', 'lab_1', 'lab_2']
		G = nx.from_pandas_edgelist(df, 'e1', 'e2')
	elif name == 'bird':
		network = open('data/aves-wildbird-network.edges','r')
		lines = network.readlines()
		network.close()
		lst = [i.strip() for i in lines]
		split = [' '.join(map(str, i.split()[:2])) for i in lst]
		G = nx.parse_edgelist(split)
	elif name == 'tortoise':
		G = nx.read_edgelist('data/reptilia-tortoise-network-fi-2011.edges')
	elif name == 'dolphin':
		G = nx.read_edgelist('data/mammalia-dolphin-florida-overall.edges')
	elif name == 'voles':
		network = open('data/mammalia-voles-plj-trapping.edges','r')
		lines = network.readlines()
		network.close()
		lst = [i.strip() for i in lines]
		split = [' '.join(map(str, i.split()[:2])) for i in lst]
		G = nx.parse_edgelist(split)
	elif name == 'uva_pre':
		network = open('data/personnetwork_exp', 'r')
		lines = network.readlines()
		lst = []
		for line in lines:
			lst.append(line.strip())
		network.close()
		H_prime = nx.parse_edgelist(lst[:6000])
		# 6000 edges is 3080 nodes
		G = H_prime.subgraph(max(nx.connected_components(H_prime))).copy()
		del lst
	elif name == 'uva_post':
		network = open('data/personnetwork_post', 'r')
		lines = network.readlines()
		lst = []
		for line in lines:
			lst.append(line.strip())
		network.close()
		H_prime = nx.parse_edgelist(lst[1000:1500])
		G = H_prime.subgraph(max(nx.connected_components(H_prime))).copy()
	elif name == 'random':
		G = nx.read_edgelist('data/random_150_0.08_12.txt')
	elif name == 'path_graph':
		G = nx.read_edgelist('data/path_graph_4.txt')
	elif name == 'uva_pool':
		network = open('data/personnetwork_pool', 'r')
		lines = network.readlines()
		lst = []
		for line in lines:
			lst.append(line.strip())
		network.close()
		G = nx.parse_edgelist(lst)


	mapping = dict(zip(G.nodes(),range(len(G))))
	graph = nx.relabel_nodes(G,mapping)
	return graph

if __name__ == "__main__":
	approx_time = time.time()

	graph = read_graph('bird')
	#graph = read_graph('path_graph')
	
	if len(sys.argv) >= 2:
		num_cascades = int(sys.argv[1])
		#n_p = int(sys.argv[1])
		#budget = int(sys.argv[2])
	else:
		num_cascades = 1000

	n_p = 4
	budget = 50

	#set_list = enumerate(graph, n_p=3)
	set_list = enumerate_random(graph, n_p=n_p, num_sets=20000)
	fl = .055
	cascade_list = cascade_construction(graph, num_cascades, fl)

	

	#with open('test_cascades/test_graph_100_0.1.pkl', 'rb') as f:
	# Make sure to update this with the probability

	#with open(f'test_cascades/uva_pool_full_graph_2000c_0.055.pkl', 'rb') as f:
	#	cascade_list = pickle.load(f)
	#cascade_list = cascade_construction(graph, num_cascades, .055)

	x, y, obj_value, variables = LinearProgram(graph, set_list, cascade_list, B=budget)
	#print(f'Objeective Value: {obj_value}')
	#print(f'Variables: {x}, {y}')


	x_prime = rounding(x)
	nonzeros = {}
	for i in x.keys():
		if x[i] > 0:
			nonzeros[i] = x[i]
	#print(f'\n\nCascades: {[i.nodes() for i in cascade_list]}\n\n')
	#print(f'LP Output: {nonzeros}, Ys: {y}')
	print(f'Sets Chosen: {nonzeros}')


	rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)

	#print(f'Total Run Time: {time.time() - approx_time} seconds ---')
	#print(f'LP Obj Val: {obj_value}, Rounded Obj Val: {rounded_obj_val}, size of x, y: {len(x)}, {len(y)}')
	with open('results_file.csv', 'a') as f:
		f.write(f'\n{len(graph)},{len(cascade_list)},{fl},{obj_value},{rounded_obj_val}, Y, {len(set_list)}, {n_p},{budget}')



