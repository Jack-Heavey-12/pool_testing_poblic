import networkx as nx
import sys
from gurobipy import *
import math
import numpy as np
import time
import pandas as pd
import json
import random



from collections import defaultdict

from matplotlib import pyplot as plt




def rounding(x, n, big_n):
	c = 2 + math.log(big_n, n+1)
	d = c * (math.log(n+1) ** 2)

	limit = np.random.uniform(0,1)

	#just a slightly cleaner way of doing the math, the other way should have still worked, though
	if limit <= np.min([1, x*d]):
		return 1
	return 0

def enumerate_random(graph, n_p=4, num_sets=1000):
	#Returns a shorter list of possible subsets for when graphs are larger
	#TODO
	nodes = list(graph.nodes())
	set_list = []
	for _ in range(num_sets):
		set_list.append(tuple(random.sample(nodes, n_p)))
	return set_list

def get_distance(dist_dict, potential_pool, max_len):
	lst = []
	for i in potential_pools:
		try:
			lst.append(dist_dict[i])
		except KeyError:
			lst.append(max_len + 1)
	return lst

def LinearProgram(graph, k, potential_pools):
	#graph is a list of sampled subgraphs, generated before being fed in here.
	#k is a fixed variable

	x = {} #defined as in the paper
	y = {} #idefined as in the paper
	M = len(graph) #number of sample graphs in list graph
	nodes = graph[0].number_of_nodes()

	src = []

	#could condense this, but want to make sure that it's the same list so it truly is random
	#could also be faster I suppose at the cost of memory
	lst_of_nodes = list(graph[0].nodes())

	#makes the same source across all samples
	#val = np.random.randint(0, nodes-1)
	for i in range(len(graph)):
		#picks a random value out of the number of nodes
		#Uncomment this line for a different source for every sample
		val = np.random.randint(0, nodes-1)
		#assigns that node as a source
		src.append(lst_of_nodes[val])


	m = Model('MinDelSS')


	
	#Defines the V_id sets in a dictionary with first key i second key is d, which will return a list of nodes at that level.
	#The way this is constructed, 
	v_set = {}
	for i in range(M):
		empt_lst = lambda:[]
		v_set[i] = defaultdict(empt_lst)
		graph[i].add_node('meta')
		graph[i].add_edge('meta', src[i])
		#This will return 1 for source nodes and the distance to the rest of them
		paths = nx.shortest_path_length(graph[i], source='meta')
		for u in potential_pools: #graph[i]:
			dists = get_distance(paths, u, nodes+1)
			min_dist = min(dists)
			v_set[i][min_dist].append(u)
			#v_set[i][paths[u]].append(u)
		graph[i].remove_node('meta')


	for i in range(M):
		for d in v_set[i].keys():
			for u in v_set[i][d]:
				if u not in x:
					#adding all of the nodes to the x dictionary - only need to add each node once despite all of the samples.
					x[str(u)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'x['+str(u)+']')
					#x[str(i)+','+str(d)+','+str(u)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'x['+str(i)+','+str(d)+','+str(u)']')
			y[str(i)+','+str(d)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'y['+str(i)+','+str(d)+']')
	m.update()

	for i in range(M):
		for d in v_set[i].keys():
			m.addConstr(quicksum(x[str(u)] for u in v_set[i][d]) >= y[str(i)+','+str(d)], name='C1_sample='+str(i)+'_dist='+str(d))
	m.update()

	'''for i in range(M):
		for d in v_set[i].keys():
			m.addConstr(quicksum(x[str(u)] for u in v_set[i][d]) <= k, name='C2_sample='+str(i)+'_dist='+str(d))
	m.update()'''

	m.addConstr(quicksum(x[str(u)] for u in list(x.keys())) <= k, name='C2_budget')


	for i in range(M):
		m.addConstr(quicksum(y[str(i)+','+str(d)] for d in v_set[i].keys()) == 1, name='C3_sample='+str(i))

	m.update()

	#Objective Function
	#First fraction is 1/N, need to somehow sum over all d values as well, think that is done in the comprehension right now but maybe need more clarity there
	m.setObjective(1/M * quicksum((y[str(i)+','+str(d)] * d) for i in range(M) for d in v_set[i].keys()), GRB.MINIMIZE)
	m.update()
	m.optimize()



	#x is the dictionary where the values of x are stored, producing the rounding here
	n = len(x)
	x_prime_dict = {}
	for j in x.keys():
		x_prime_dict[j] = rounding(x[j].x, n, num_samples)


	#producing the number of infections here by saving the length of the connected component to the selected source
	infection_list = []
	for i in range(M):
		infection_list.append(len(nx.node_connected_component(graph[i], src[i])))



	#Want to calculate the objective value now for the rounded values
	#first, create a smaller set which has all of the nodes where they are in s_r
	s_r = set()
	for i in list(x_prime_dict.keys()):
		if x_prime_dict[i] == 1:
			s_r.add(i)


	obj_vals = []
	for i in range(M):
		paths = nx.shortest_path_length(graph[i], source=src[i])
		shortest = nodes
		for u in list(s_r):
			
			dists = get_distance(paths, s_r, nodes+1)
			min_dist = min(dists)

			if min_dist < shortest:
				shortest = min_dist
		obj_vals.append(shortest + 1)



	#Return stuff here, so want to calculate everything first so that we can return everything with just one function call


	return m, m.objVal, x, x_prime_dict, np.mean(infection_list), np.mean(obj_vals), y, src, v_set


#Function for sampling edges with probability p. Will return a list of sample graphs.
def sampling(num_samples, graph, p):

	lst = []

	#Generates one graph per sample
	for i in range(num_samples):
		TempGraph = nx.Graph()

		TempGraph.add_nodes_from(graph.nodes())

		#Adds an edge if it is randomly selected with probability p
		for j in graph.edges():
			r = np.random.random()

			if r <= p:
				TempGraph.add_edge(j[0], j[1])

		lst.append(TempGraph)
		print(i)
	return lst



def produce_plot(input_name, output_string):
	df = pd.read_csv(input_name)

	df['Percent'] = df['Rounded X Value'] / df['Value of K']

	fig = plt.plot(df['Value of K'], df['Value of K'], color='r', marker='o')

	plt.plot(df['Value of K'], df['Rounded X Value'], color='b', marker='o')
	plt.legend(['Value of Budget K', 'Size of Selected Set S_r'])
	plt.title('Violation of Budget Constraints')

	plt.savefig(output_string + '.png')
	plt.clf()

	fig2 = plt.plot(df['Value of K'], df['Percent'], 'o')
	plt.title('Violation of K as a Percentage of K')
	plt.savefig(output_string +'_percent.png')
	plt.clf()


	fig3 = plt.plot(df['Rounded X Value'], df['new_objVal'], color='m', marker='o')
	plt.title('Size of Sensor Set vs. Rounded Mean Detection Time')
	plt.savefig(output_string + '_objvK.png')
	plt.clf()


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

#main function
if __name__ == '__main__':
	main = False
	if len(sys.argv) > 1:
		dataset = sys.argv[1] #input the original dataset/graph name
		k = int(sys.argv[2])

		main = True

	outputfile = 'pool_output' #change the output file name here
	
	#G = []
	paths = []
	levels = [] 
	sources = []
	inedges = []
 
	print("Generating Simulations")
	
	#Change num samples here
	#numSamples = 20
	
	#Change probability of transmission here
	#Used to determine the probability of a given edge being sampled
	p = 0.1

	H = read_graph('bird')



	nodes = len(H.nodes()) #small n

	k_arr = []
	obj_val_lst = []
	x_prime_dict_arr = []
	inf_list_lst = []
	new_obj_val_lst = []

	#prob_lst = [.001, .005, .01, .03, .05, .1, .15, .2, .25, .3, .35, .4]
	if not main:
		num_samples = 2500
		#H = nx.parse_edgelist(dataset)
		G = sampling(num_samples, H, p)
		potential_pools = enumerate_random(H)
		k = 10

		model, obj_val, x_dict, x_prime_dict, inf_list, new_obj_val, y_vals, src, v_set = LinearProgram(G, k, potential_pools)

		with open('delay_results.csv', 'a') as f:
			f.write(f'\n{obj_val},{new_obj_val}')
		print(f'Results: {obj_val}, {new_obj_val}')


	else:
		for i in range(25, 50, 25):
		#for i in prob_lst:
			num_samples = 2500
			k = 100 #VALUE_FOR_K
			#k = np.floor((i/1000) * nodes)
			#p = i
			G = sampling(num_samples, H, p)

			#UPDATE - Deleted LIST_OF_SOURCES as an argument, defined in function randomly.
			model, obj_val, x_dict, x_prime_dict, inf_list, new_obj_val, y_vals, src, v_set = LinearProgram(G, k)


			#NEED TO EVALUATE THIS WRITE STATEMENT BELOW HERE
			k_arr.append(k)
			x_prime_dict_arr.append(sum(x_prime_dict.values()))
			inf_list_lst.append(inf_list)
			new_obj_val_lst.append(new_obj_val)
			obj_val_lst.append(obj_val)


			textfile = open('k_values_post_p15.csv', 'w')
			textfile.write('Value of K,Rounded X Value,objVal,new_objVal,Number of Infections (mean)\n')
			for val in range(len(k_arr)):
				textfile.write(str(k_arr[val])+','+str(x_prime_dict_arr[val])+','+str(obj_val_lst[val])+
					','+str(new_obj_val_lst[val])+','+str(inf_list_lst[val])+'\n')
			textfile.close()


			'''y_file = open('y_vals.csv', 'w')
			y_file.write('Var_name,Optimized_Value\n')
			for val in list(y_vals.keys()):
				y_file.write(str(val)+','+str(y_vals[val].x)+'\n')
			y_file.close()'''

			'''x_vals = open('x_vals_post.csv', 'w')
			x_vals.write('Var_name,Optimized_Value\n')
			for val in list(x_dict.keys()):
				x_vals.write('x['+str(val)+'],'+str(x_dict[val].x)+'\n')
			x_vals.close()'''

			'''v_file = open('v_file.tsv', 'w')
			v_file.write('Sample\tDistance\tSource\tValues\n')
			for i in range(len(G)):
				for d in list(v_set[i].keys()):
					v_file.write(str(i)+'\t'+str(d)+'\t'+str(src[i])+'\t'+str(v_set[i][d])+'\n')
			v_file.close()'''



		#CAN CHANGE WHAT YOU WANT THE PLOT TO BE CALLED HERE
		#produce_plot('k_values_post_p15.csv', 'post_covid_p15')


	print('Done!')

	