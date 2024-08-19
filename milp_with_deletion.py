import networkx as nx
import sys
import math
import numpy as np
import time
import pandas as pd
import json
import random

import pickle


from collections import defaultdict
import itertools


# Solving the problem min 1/N \sum_{v,i} z(v,i)
# S.T. for all v not in C(src, i): z(v,i) + \sum_{S in F(v,i)} x(S) >= 1
#               \sum_{S) x(S) <= B


# Definint functions as defined in https://arxiv.org/pdf/cs/0205039
def lmin(y): #y is a collection of real values y_1...y_n
	return -1 * np.log(sum([np.e ** (-i) for i in y]))

def lmax(y):
	return np.log(sum([np.e ** i for i in y]))

def partial_j(M, x, j): #M is an i x j matrix, x is a result variable, j is a given row
	top = sum([M[i][j] * np.e ** ((M @ x)[i]) for i in list(range(len(M)))])
	bottom = sum([np.e ** ((M @ x)[i]) for i in list(range(len(M)))])
	return top / bottom


def enumerate(graph, n_p=3):
	#returns list of the possible subsets that can be chosen
	#NOTE: This list is O(n^{n_p}), be aware of memory constraints
	nodes = list(graph.nodes())
	#set_list = []
	set_list = set(itertools.combinations(nodes, n_p))
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
		#       Hopefully shouldn't be any assignment issues either

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

def define_matrices(pools, nodes, cascades, budget):
	# below product allows the cascade to be referenced by an integer, which we can use to index into the F_i dict
	approx_time = time.time()
	v_i_list = list(itertools.product(nodes, list(range(len(cascades)))))


	#v_i_list_masked = list(itertools.product(list(range(len(nodes))), list(range(len(cascades)))))

	v_i_len = len(v_i_list)
	print(v_i_len)
	pool_len = len(pools)
	casc_len = len(cascades)

	# x_vector will be z(v,i)s first then the x(s) second
	C = np.zeros((pool_len + v_i_len, v_i_len), dtype=np.dtype('uint8'))

	approx_time = time.time()

	F_i = {} # Superset of F_vi, will define the sets here for each cascade. From here, only have to check for membership for the later constraint
	for i in range(len(cascades)):
		F_i[i] = set([S for S in set_list if not any(x in cascades[i] for x in S)])


	for i in range(pool_len):
		#A[i+v_i_len, :] = np.array([0] * v_i_len)
		#A[i+v_i_len, :] = np.array([1 if (x not in casc[j]) for x in nodes for j in range(casc_len)])
		C[i+v_i_len, :] = np.array([1 if all([x in pools[i], pools[i] in F_i[j]]) else 0 for (x, j) in v_i_list])

	C_vid = np.identity(v_i_len, dtype=np.dtype('uint8'))

	for i in range(v_i_len):
		C[i] = C_vid[i, :]

	#NOTE: Have to transpose C because I did this backwards
	C = C.T

	c = np.array([0 if x in cascades[j] else 1 for (x, j) in v_i_list])
	# moving this here as we don't need to perform this delete operation every run of the approximation algorithm

	# So now have C @ x >= c

	

	possible_cleared_nodes = set()
	for i in range(len(cascades)):
		for v in nodes:
			val =  v in cascades[i]
			if not val:
				possible_cleared_nodes.add((v,i))



	P = np.array([[0] * v_i_len + [1] * pool_len, 
				[1 if (v,i) in possible_cleared_nodes else 0 for (v,i) in v_i_list] + [0] * pool_len])
	# P is just a one dimensional vector since we only have one packing constraint, so have P @ x <= p
	#p = budget

	print(f'All Matrices Construction Time: {time.time() - approx_time} seconds ---')
	print(f'Total time spent so far: {time.time() - begin_time} seconds ---')
	return C, c, P, #p Not returning little p because it depends on lambda, which will change each iteration, other things won't.



'''def Approximation(graph, set_list, cascade_list, C, c, P, budget, lam=20, ep=.05):
	# Defining this based on Algorithm 1 of Young paper
	# m is the number of constraints, so len(A) + 1. In this case, we have (v * i) + 1 constraints
	p = np.array([budget],[lam])

	v_i_list = list(itertools.product(list(graph.nodes()), list(range(len(cascade_list)))))
	
	v_i_len = len(v_i_list)
	pool_len = len(set_list)
	casc_len = len(cascade_list)

	x_0 = np.zeros(v_i_len + pool_len).reshape(-1, 1)

	# this first definition holds because p only has one row
	U = (P @ x_0 + np.log(len(C) + 1)) / ep**2
	u = np.max(U)
	p_x = np.array([1 + ep]).reshape(1,1) ** (P @ x_0)

	# Operations to create c_x vector
	temp = C @ x_0
	c_x = np.array([1 - ep] * len(C))
	c_x = np.array([c_x[i] ** temp[i] for i in list(range(len(C)))]) #something is wrong with this line, not worrying about it for now
	#c_x = np.array([1 - ep] * len(C)) (C @ x_0)
	#print(f'shape before exponent: ')
	#print(f'u: {u}, c_x shape: {c_x.shape}, C @ x_0 shape: {(C @ x_0).shape}')
	c_x = np.array([i if i <= u else 0 for i in c_x])

	p_x_norm = np.sum(p_x)
	c_x_norm = np.sum(c_x)
	#print(f'P.T shape: {P.T.shape}, p_x shape: {p_x.shape}, C.T shape: {C.T.shape}, c_x shape: {c_x.shape}')

	lam_x_j = (P.T.reshape(len(P.T), 1) @ p_x) / (C.T @ c_x)
	lam_star = np.min(np.sum(lam_x_j, axis=1))

	lam_0 = p_x_norm / c_x_norm
	# Finished all of the definitions here, should work out
	x = x_0

	#repeatedly iterate over the two operations until done, right? So should just do a while true here?
	while True:
		# Operation (a): TODO
		if lam_star <= (1 + 4*ep) * lam_0:
			#TODO
			#delta = np.array([0 if lam_x_j[i] >= (1 + 4*ep)lam_0[i] else 0.5 for i in range(len(lam_x_j))])
			delta = np.array([0] * len(lam_x_j))
			#ensures the maximum part of delta related to the Ps is 1 (I think)?
			delta_p = np.array([0] * v_i_len + [1/pool_len] * pool_len)
			
			delta_C = np.zeros(C.shape)
			
			for i in range(len(C)):
				#delta_C[i] = np.array([i if delta_C[i][j] == 0 else 1/np.sum(delta_C[i]) for j in range(len(delta_C[i]))])
				mat_mul = C[i] @ x
				delta_C[i] = np.array([np.inf if mat_mul >= u else 1/np.sum(C[i]) for _ in list(range(len(C[i])))])

			#Pretty sure that this will make sure that it works!!
			#full_delta = np.min(np.array(enumerate(np.vstack(delta_C, delta_p)), key=lambda x: x[1] if x[1] > 0 else 0), axis=0)
			
			full_delta = np.min(np.vstack((delta_C, delta_p)), axis=0).reshape(-1, 1)
			print(f'full_delta shape: {full_delta.shape}, shape of x: {x.shape}')
			for i in range(len(full_delta)):
				if lam_x_j[i] <= (1 + 4*ep) * lam_0:
					full_delta[i] = 0
			x += full_delta
				
			if np.min(C @ x) >= u:
				return x / u



		# Operation (b)
		elif lam_star >= (1 + ep) * lam_0:
			lam_0 = (1 + ep) * lam_0
'''

def calculate_top(np_arr):
	return np.ceil(np.log2(np_arr))


def Approximation2(graph, set_list, cascades, C, c, P, budget, lam=20, ep=0.25):
	#based off of Algorithm 2 in the Young paper

	approx_start_time = time.time()

	v_i_list = list(itertools.product(list(graph.nodes()), list(range(len(cascades)))))


	#v_i_list_masked = list(itertools.product(list(range(len(nodes))), list(range(len(cascades)))))
	
	v_i_len = len(v_i_list)
	pool_len = len(set_list)
	casc_len = len(cascades)

	p = np.array([budget, lam]).T

	P = P / p[:,None]
	mask = ~(c > 0)
	C = np.delete(C, mask, axis=0)
	#print(f'C after delete: {C}')
	print(f'Shape of C: {C.shape}')
	#C = np.where()

	x =  np.array([0] * (v_i_len + pool_len), dtype='float64') ; lam_0 = len(P) / len(C) #P.shape[0] / C.shape[0] ; #number of packing constraints, covering constraints
	U = np.log(C.shape[0] + P.shape[0]) / (ep ** 2)

	P_hat = np.array((P @ x)) #- np.array([1/2] * len(P)) #don't need this, as is guaranteed by the proof later on since we are outside of the while loop
	#p_hat_little = np.array([1+ep] * len(P_hat))
	#p_hat_little = np.array([(1+ep) ** P_hat[i] for i in list(range(len(P_hat)))], dtype='float64')
	p_hat_little = (1+ep) ** P_hat
	
	C_hat = (C @ x) #- np.array([1/2] * len(C)) #similar to line 178
	#c_hat_little = np.array([(1 - ep) ** C_hat[i] if C_hat[i]<=U else 0 for i in list(range(len(C_hat)))], dtype='float64')
	c_hat_little = (1-ep) ** C_hat #don't need to worry about the 0 condition intially, not possible due to 0s in x


	itera=0
	#min_ratio_j = np.inf
	while True: #this while loop is here to cover for the "Repeat" in line 5
		print(f'X-shape: {x.shape}')
		ratio_js = []
		for j in range(len(x)):

			#print(f'j: {j}')
			lam_j_hat = (P.T[j] @ p_hat_little) / (C.T[j] @ c_hat_little)
			#print(f'lam_j_hat: {lam_j_hat}, other side: {(1+ep) ** 2 * lam_0 / (1-ep)}')
			it = 0
			# This is calculating the values to ensure that a given value is eligible to be incremented by inc.
			top_arr_P = 0.5 ** np.reciprocal(np.ceil(np.log2(P.T[j]))) ; top_arr_C = 0.5 ** np.reciprocal(np.ceil(np.log2(C.T[j]))) ;
			while lam_j_hat <= (1+ep) ** 2 * lam_0 / (1-ep):
				it += 1
				
				top = 1/2**it

				# computing the maximum while working to only do the matrix multiplication once
				vec = (C @ x) ; vec = [C.T[j][i] if vec[i] <= U else 0 for i in range(C.shape[0])];
				val = max(np.max(P.T[j]), max(vec))
				#val = max(max(P.T[j]), max([C.T[j][i] if C[i] @ x <= U else 0 for i in range(C.shape[0])], default=np.nan))
				if val == 0:
					continue
				inc = .5 * val
				x[j] = x[j] + inc
				

				#P_hat = np.where(np.logical_and(P.T[j] > 0, top_arr_P < inc), P_hat + val*P.T[j], P_hat)
				inds_to_update = np.where(np.logical_and(P.T[j] > 0, top_arr_P < inc))[0]
				P_hat[inds_to_update] += val*P.T[j][inds_to_update]

				# Second condition should never happen in our program, but for completeness sake
				#C_hat = np.where(np.logical_and(C.T[j] > 0, top_arr_C < inc), C_hat + val*C.T[j], C_hat)
				inds_to_update_2 = np.where(np.logical_and(C.T[j] > 0, top_arr_C < inc))[0]
				C_hat[inds_to_update_2] += val*C.T[j][inds_to_update_2]

				# Don't think the below is really different because the tops should *basically* all be in the same equivalence group, but 
				#P_hat = np.where(P.T[j]*val > top, P_hat + val*P.T[j], P_hat)
				#C_hat = np.where(C.T[j]*val > top, C_hat + val*C.T[j], C_hat)

				# should be able to use np.where for all of these since we only need to change where it is incrementing
				# Should be way faster than below

				#mask = np.where(C_hat >= U)[0]
				mask = (C_hat >= U)
				if np.any(mask):
					C = np.delete(C, mask, axis=0)
					C_hat = np.delete(C_hat, mask, axis=None)
					c_hat_little = np.delete(c_hat_little, mask, axis=None)

				p_hat_little[inds_to_update] = (1+ep) ** P_hat[inds_to_update]

				inds_to_update_2 = np.where(np.logical_and(C.T[j] > 0, top_arr_C < inc))[0]
				c_hat_little[inds_to_update_2] = (1-ep) ** C_hat[inds_to_update_2]
				#p_hat_little = np.where(P.T[j] > 0, (1+ep) ** P_hat, p_hat_little)
				#c_hat_little = np.where(C.T[j] > 0, (1 - ep) ** C_hat, c_hat_little)
				#c_hat_little = np.where(C_hat <= U, c_hat_little, 0)

				#p_hat_little = np.array([(1+ep) ** P_hat[i] for i in list(range(len(P_hat)))])
				#c_hat_little = np.array([(1 - ep) ** C_hat[i] if C_hat[i]<=U else 0 for i in list(range(len(C_hat)))])
				
				
				
				

				lam_j_hat = (P.T[j] @ p_hat_little) / (C.T[j] @ c_hat_little) # this is the second part of lines 12, 13
				

				

				#if it % 25 == 0:
					#print(f'for x[{j}], Iteration {it} in while-loop')
				if it > 7500:
					print('System exiting')
					print(f'Current C: {C}')


					sys.exit()
				print(f'Iteration {it}: Should be number of iterations O(mU)')
			# not sure if this is the right spot for this, but we ride?
			mask = (C_hat >= U)
			if np.any(mask):
				C = np.delete(C, mask, axis=0)
				C_hat = np.delete(C_hat, mask, axis=None)
				c_hat_little = np.delete(c_hat_little, mask, axis=None)

			#print(f'Exit condition: {C_hat.size == 0}, C_hat size: {C_hat.size}')
			if C_hat.size == 0:
				print(f'Approximation time length: {time.time() - approx_start_time} seconds')
				return x / U

			#checking feasibility condition from https://arxiv.org/pdf/cs/0205039
			#print(f'Partial P: {partial_j(P, x/U, j)}, Partial C: {partial_j(C, -1*(x/U), j)}')
			try:
				ratio_js.append(partial_j(P, x/U, j) / partial_j(C, -1*(x/U), j))
			except ZeroDivisionError:
				ratio_js.append(np.inf)

		if min(ratio_js) > 1:
			print(f'Current x: {x/U}\n J: {j}, Current C_column: {C.T[j]}, Current P_column: {P.T[j]}\n Exiting Requirement: {C_hat.size}')
			return 'infeasible!'
				#pass



			#print(f'Value of x[{j}]: {x[j]}, ')
		# Operation 2
		lam_0 *= (1+ep)
		itera += 1

		print(f'End While Loop Iteration {itera}, Current lam_0 max: {lam_0}, remaining shape of C: {C.shape}')

def calculate_E_welfare(x_prime, cascade_list):
	#TODO
	num_cascades = len(cascade_list)
	running_clearances = 0
	for S in x_prime.keys():
		#x_prime.keys() is going to be a set, which means have to iterate through the set and then each cascade to see if it's cleared.
		for v in S:
			for i in cascade_list:
				#will give a 1 if the node is cleared in that cascade, note we DON'T want it in the connected component
				val = int(not (v in i))
				running_clearances += val
	return running_clearances / num_cascades


def rounding(x_dict):
	#Will return the rounded values in a dictionary, which correspond to specific subsets to choose
	#Doing this in a separate function in case the rounding function changes later

	x_prime_dict = {}
	for S in x_dict.keys():
		limit = np.random.uniform(0,1)
		#print(type(x[S]))
		#print(dir(x[S]))
		#sys.exit()
		if limit <= x_dict[S]:
			x_prime_dict[S] = 1
	#NOTE:  This will not have *every* set in this dictionary, only the pools that we are going to end up choosing
	#       Figure this is easier than trying to check if every value is one or zero, we can jsut compare length for expectation, etc.

	#NOTE2: The keys in x, and thus x_prime_dict, are the sets themselves. Think that's the best way to do it, shouldn't have any assignment issues?
	return x_prime_dict



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
	global begin_time 
	begin_time = time.time()


	graph = read_graph('bird')
	start_time = time.time()
	
	if len(sys.argv) >= 2:
		ep = float(sys.argv[1])
	else:
		ep = .1

	num_sets = 3000

	n_p = 4
	budget = 50

	set_list = enumerate_random(graph, n_p=n_p, num_sets=num_sets)

	fl = .055
	#with open('test_cascades/fast_gnp_10c_0.33p.pkl', 'rb') as f:
	#   cascade_list = pickle.load(f)

	cascade_list = cascade_construction(graph, 202, fl)

	C, c, P = define_matrices(set_list, list(graph.nodes()), cascade_list, budget)

	initial_lam = len(graph) * len(cascade_list)
	condition = False

	it = 0
	while not condition:
		it += 1
		x = Approximation2(graph, set_list, cascade_list, C, c, P, budget, lam=initial_lam)
		print(f'Current x: {x}')
		if isinstance(x, str):
			condition = True
			print(f'Got to infeasible with iteration {it}, lambda {initial_lam}')
		else:
			old_x = x
			initial_lam /= 2
		print(f'Current Lambda: {initial_lam}')
		if initial_lam < 1e-200:
			print(f'Initial lam got too small on iteration {it}')
			condition = True

	print(old_x)

	x_dict = {}
	true_xs = [i for i in old_x[-len(set_list):] if i > 0 ] # [0, 1], \{0,1\}
	for i in range(len(true_xs)):
		x_dict[set_list[i]] = true_xs[i]

	true_zs = old_x[:len(x) - len(set_list)]

	print(f'Sum of x: {sum(true_xs)}')



	#NOT IMPLEMENTED YET
	x_prime = rounding(x_dict)

	#print(f'\n\nCascades: {[i.nodes() for i in cascade_list]}\n\n')
	#print(f'LP Output: {nonzeros}, Ys: {y}')
	#print(f'Sets Chosen: {x_prime}')


	rounded_obj_val = calculate_E_welfare(x_prime, cascade_list)


	#print(f'Result: {x}, max value: {x.max()}, sum: {sum(x)}')
	print(f'Rounded Objective Value: {rounded_obj_val}, Sets chosen: {x_prime}')
	print(f'Total Execution Time: {time.time() - start_time} seconds ---')

	with open('milp_results.csv', 'a') as f:
		f.write(f'\n{ep},{rounded_obj_val}')

	'''x_prime = rounding(x)
	nonzeros = {}
	for i in x.keys():
		if x[i] > 0:
			nonzeros[i] = x[i]
	print(f'Sets Chosen: {nonzeros}')'''
