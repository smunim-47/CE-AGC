

import cvxpy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy.linalg as la
import random

def create_regular_bipartite_graph(n1, n2, d1, d2):
    if n1 * d1 != n2 * d2:
        raise ValueError("No d-regular bipartite graph possible with these parameters")

    left_nodes = list(range(n1))
    right_nodes = list(range(n1, n1 + n2))

    B = nx.Graph()
    B.add_nodes_from(left_nodes, bipartite=0)
    B.add_nodes_from(right_nodes, bipartite=1)

    edges = []
    left_degrees = {node: 0 for node in left_nodes}
    print(left_degrees)
    right_degrees = {node: 0 for node in right_nodes}

    # Shuffle the nodes to randomize connections
    random.shuffle(left_nodes)
    random.shuffle(right_nodes)

    # Create a list of edge candidates
    edge_candidates = [(u, v) for u in left_nodes for v in right_nodes]
    print(len(edge_candidates))
    print(edge_candidates)
    random.shuffle(edge_candidates)
    print(edge_candidates)

    i = 1
    while(i):
        random.shuffle(edge_candidates)
        print('edge_candidtates', edge_candidates)
        for u, v in edge_candidates:
            print(u,v)
            if left_degrees[u] < d1 and right_degrees[v] < d2:
                B.add_edge(u, v)
                left_degrees[u] += 1
                right_degrees[v] += 1
                if all(left_degrees[node] == d1 for node in left_nodes) and all(right_degrees[node] == d2 for node in right_nodes):
                    break

        if (all(left_degrees[node] == d1 for node in left_nodes) and all(right_degrees[node] == d2 for node in right_nodes)):
           print('Huh')
           print(left_degrees)
           print(right_degrees)
           i = 0
        else:
          print('nope')
          print(B.edges())
          B.remove_edges_from(list(B.edges()))
          print(B.edges())
          left_degrees = {node: 0 for node in left_nodes}
          print(left_degrees)
          right_degrees = {node: 0 for node in right_nodes}


    return B

def incidence_matrix(B, n1, n2):
    matrix = np.zeros((n1, n2), dtype=int)
    print(B.edges())
    print(len(B.edges()))

    for u, v in B.edges():
        if u < n1:
            matrix[u, v - n1] = 1
        else:
            matrix[v, u - n1] = 1
    return matrix

n_1 = 20
n_2 = 40
d_1 = 6
d_2 = 3

B = create_regular_bipartite_graph(n_1, n_2, d_1, d_2)
A = incidence_matrix(B, n_1, n_2)






np.random.seed(43)
mean = 0 # Desired mean
std_dev = 1  
v1 = np.random.normal(loc=mean, scale=std_dev, size=(n_2, 1))


v1 = np.reshape(v1, (v1.shape[0],1))


P = np.zeros((n_1,n_2))
Q = np.zeros((n_1,n_2))
for k in range(A.shape[0]):
    nz = np.nonzero(A[k,:])
    # print(nz)
    # print(v1[nz])
    P[k,:][nz] = np.reshape(v1[nz], (len(nz[0]),))/ (la.norm(v1[nz]))**2
   

U, S, Vt = np.linalg.svd(P)
v2 = np.reshape(Vt.T[:, -1], (Vt.T[:, -1].shape[0],1))
Q = np.zeros((n_1,n_2))
for k in range(A.shape[0]):
    nz = np.nonzero(A[k,:])
    # print(nz)
    # print(v1[nz])
    Q[k,:][nz] = np.reshape(v2[nz], (len(nz[0]),))/ (la.norm(v2[nz]))**2

E = np.vstack((P,Q))



v1= np.ones((n_2,1))
P1 = np.zeros((n_1,n_2))

for k in range(A.shape[0]):
    nz = np.nonzero(A[k,:])
    # print(nz)
    # print(v1[nz])
    P1[k,:][nz] = np.reshape(v1[nz], (len(nz[0]),))/ (la.norm(v1[nz]))**2
    

z = cp.Variable((n_2, 1), boolean=True)
v2 = 2*z-1
constraints = [P1@v2==0]
objective = cp.Minimize(10)
problem = cp.Problem(objective, constraints)
problem.solve()




if problem.status == cp.OPTIMAL:
  print("Okay")




#print("Optimal P:", P.value)
#print("Optimal P rounded:", P_rounded)0.

else:
  print("Problem is infeasible or unbounded")
v2 = v2.value
Q1 = np.zeros((n_1,n_2))
for k in range(A.shape[0]):
    nz = np.nonzero(A[k,:])
    # print(nz)
    # print(v1[nz])
    Q1[k,:][nz] = np.reshape(v2[nz], (len(nz[0]),))/ (la.norm(v2[nz]))**2



E1 = np.vstack((P1,Q1))

E2 = np.vstack((A,A))


np.random.seed(42)
D1 = np.diag(np.random.choice([1, -1], size= (n_2), p=[0.5, 0.5]))
D2 = np.diag(np.random.choice([1, -1], size= (n_2), p=[0.5, 0.5]))

E3 = np.vstack((A@D1,A@D2))

m = 2
F = np.kron(np.eye(m), np.ones((n_1,1)))
p = 15
n_w = E.shape[1]
n_p = n_1
s_list= np.linspace(0, ((.25*n_w)), p + 1, dtype=int)
# s_list = m*(s_list)
frac_stragglers = s_list/n_w
itr = 1000
ones = np.ones((n_1,1))

opt_err_array = np.zeros((len(s_list), itr))
opt_max = np.zeros(len(s_list))
opt_min = np.zeros(len(s_list))
opt_avg = np.zeros(len(s_list))

opt_err_array_1 = np.zeros((len(s_list), itr))
opt_max_1 = np.zeros(len(s_list))
opt_min_1 = np.zeros(len(s_list))
opt_avg_1 = np.zeros(len(s_list))

opt_err_array_2 = np.zeros((len(s_list), itr))
opt_max_2 = np.zeros(len(s_list))
opt_min_2 = np.zeros(len(s_list))
opt_avg_2 = np.zeros(len(s_list))


opt_err_array_3 = np.zeros((len(s_list), itr))
opt_max_3 = np.zeros(len(s_list))
opt_min_3 = np.zeros(len(s_list))
opt_avg_3 = np.zeros(len(s_list))


b_opt_err_array = np.zeros((len(s_list), itr))
b_opt_max = np.zeros(len(s_list))
b_opt_min = np.zeros(len(s_list))
b_opt_avg = np.zeros(len(s_list))


b_opt_err_array_1 = np.zeros((len(s_list), itr))
b_opt_max_1 = np.zeros(len(s_list))
b_opt_min_1 = np.zeros(len(s_list))
b_opt_avg_1 = np.zeros(len(s_list))

for i in range(len(s_list)):
    for j in range(itr):
            #print(i,j)
            np.random.seed(78+j)
            # random_non_stragglers = np.random.choice(np.arange(0, n_w), int(n_w-s_list[i]), replace=False)
            random_stragglers = np.random.choice(np.arange(0, int(n_w)), int(s_list[i]), replace=False)
            random_non_stragglers = np.setdiff1d(np.arange(0, n_w),random_stragglers)
            random_non_stragglers = np.sort(random_non_stragglers)
            E_F = E[:, random_non_stragglers]
            E_F_1 = E1[:, random_non_stragglers]
            E_F_2 = E2[:, random_non_stragglers]
            E_F_3 = E3[:, random_non_stragglers]
            P_F = P[:, random_non_stragglers]
            Q_F = Q[:, random_non_stragglers]
            P_F_1 = P1[:, random_non_stragglers]
            Q_F_1 = Q1[:, random_non_stragglers]
            # R_F = R[:, random_non_stragglers]
            # E_F_3 = E3[:, random_non_stragglers]
            #print(random_non_stragglers.shape)
            #print(random_non_stragglers)
            #print(E_F.shape)
            #print(E_F_1.shape)
            opt_err = 0
            opt_err_1 = 0
            opt_err_2 = 0
            opt_err_3 = 0

            for t in range(m):

                      a = F[:,t]
                      a = np.reshape(a, (m*n_p,1))

                      #print(a)
                      # print(a.conj().T@a-a.conj().T@E_F@ np.linalg.inv(E_F.conj().T@E_F).conj().T@E_F.conj().T@a)



                      #opt_err = opt_err + a.T@a-a.T@E_F@ np.transpose(np.linalg.inv(np.transpose(E_F)@E_F))@E_F.T@a
                      # s1 = a.conj().T@a-a.conj().T@E_F@ np.linalg.inv(E_F.conj().T@E_F).conj().T@E_F.conj().T@a
                      # opt_err = opt_err + a.conj().T@a-a.conj().T@E_F@ np.linalg.inv(E_F.conj().T@E_F).conj().T@E_F.conj().T@a
                      x, residuals, rank, s = np.linalg.lstsq(E_F, a, rcond=None)
                      opt_err = opt_err + np.linalg.norm(E_F@x-a,2)**2
                      x, residuals, rank, s = np.linalg.lstsq(E_F_1, a, rcond=None)
                      opt_err_1 = opt_err_1 + np.linalg.norm(E_F_1@x-a,2)**2
                      x, residuals, rank, s = np.linalg.lstsq(E_F_2, a, rcond=None)
                      opt_err_2 = opt_err_2 + np.linalg.norm(E_F_2@x-a,2)**2
                      x, residuals, rank, s = np.linalg.lstsq(E_F_3, a, rcond=None)
                      opt_err_3 = opt_err_3 + np.linalg.norm(E_F_3@x-a,2)**2
                      #print(residuals)
                      # opt_err_2 = opt_err_2 + a.conj().T@a-a.conj().T@E_F_2@ np.linalg.inv(E_F_2.conj().T@E_F_2).conj().T@E_F_2.conj().T@a



                      # print('norm')
                      # print(np.linalg.norm(E_F.conj().T@a, 2))
                      # print('eig')
                      # print(np.sort(np.linalg.eigvals(E_F.T@E_F))[-1])
                      # print(((np.linalg.norm(E_F.conj().T@a, 2)**2 )/np.sort(np.linalg.eigvals(E_F.T@E_F))[-1]))
                      #print(np.sort(np.linalg.eigvals(E_F.T@E_F))[-1])
                      #opt_err_1 = opt_err_1 + a.conj().T@a-a.conj().T@E_F_1@ np.linalg.inv(E_F_1.conj().T@E_F_1).conj().T@E_F_1.conj().T@a
                      # x, residuals, rank, s = np.linalg.lstsq(E_F_2, a, rcond=None)
                      # print(residuals)
                      # # opt_err_2 = opt_err_2 + a.conj().T@a-a.conj().T@E_F_2@ np.linalg.inv(E_F_2.conj().T@E_F_2).conj().T@E_F_2.conj().T@a
                      # if residuals.size > 0:
                      #   opt_err_2 = opt_err_2 + residuals[0]
                      # else:
                      #     opt_err_2 = opt_err_2 + 0


                      # opt_err_3 = opt_err_3 + a.conj().T@a-a.conj().T@E_F_3@ np.linalg.inv(E_F_3.conj().T@E_F_3).conj().T@E_F_3.conj().T@a
                      #print(np.imag(opt_err))

            #print(opt_err.item())
            opt_err_array[i,j] = (np.real(opt_err))
            opt_err_array_1[i,j] = (np.real(opt_err_1))
            opt_err_array_2[i,j] = (np.real(opt_err_2))
            opt_err_array_3[i,j] = (np.real(opt_err_3))
            
            tilde_p = P_F.T@P_F + Q_F.T@Q_F
            tilde_A =  np.diag(np.diag(tilde_p) + np.sum(np.abs(tilde_p), axis=1) - np.abs(np.diag(tilde_p)))


            b_opt_err_array[i,j] = m*n_p - ones.T@P_F@la.inv(tilde_A)@P_F.T@ones - ones.T@Q_F@la.inv(tilde_A)@Q_F.T@ones
            tilde_p = P_F_1.T@P_F_1 + Q_F_1.T@Q_F_1
            tilde_A =  np.diag(np.diag(tilde_p) + np.sum(np.abs(tilde_p), axis=1) - np.abs(np.diag(tilde_p)))
            b_opt_err_array_1[i,j] = m*n_p - ones.T@P_F_1@la.inv(tilde_A)@P_F_1.T@ones - ones.T@Q_F_1@la.inv(tilde_A)@Q_F_1.T@ones
            



    opt_max[i] = np.max(opt_err_array[i,:])
    opt_min[i] = np.min(opt_err_array[i, :])
    opt_avg[i] = np.mean(opt_err_array[i, :])

    opt_max_1[i] = np.max(opt_err_array_1[i,:])
    opt_min_1[i] = np.min(opt_err_array_1[i, :])
    opt_avg_1[i] = np.mean(opt_err_array_1[i, :])

    opt_max_2[i] = np.max(opt_err_array_2[i,:])
    opt_min_2[i] = np.min(opt_err_array_2[i, :])
    opt_avg_2[i] = np.mean(opt_err_array_2[i, :])


    opt_avg_3[i] = np.mean(opt_err_array_3[i, :])

    b_opt_max[i] = np.max(b_opt_err_array[i,:])
    # b_opt_min[i] = np.min(b_opt_err_array[i, :])
    b_opt_avg[i] = np.mean(b_opt_err_array[i, :])
    b_opt_avg_1[i] = np.mean(b_opt_err_array_1[i, :])
    b_opt_max_1[i] = np.max(b_opt_err_array_1[i, :])

b_opt_max = np.zeros(len(s_list))

for i in range(b_opt_max.shape[0]):
 b_opt_max[i] = np.max(b_opt_err_array[i])

new_lower_bound = np.zeros((len(s_list)))
lb = np.zeros((m))
for i in range(len(s_list)):
  for j in range(m):
      lb[j] = np.floor((n_p*(s_list[i] + m - j -1))/(n_w*d_2)) * (j+1)
  new_lower_bound[i] = np.max(lb)




plt.plot(frac_stragglers, b_opt_max_1, color = 'green', marker ='o', label = 'Upp. Bd. $\pm 1$ Cons.')
plt.plot(frac_stragglers, opt_avg_1, color = 'blue', marker ='^', label = 'Avg. Err. $\pm 1$ Cons.')
plt.plot(frac_stragglers, b_opt_max, color = 'black', marker ='H', label = 'Upp. Bd. Gauss. Cons.')
plt.plot(frac_stragglers, opt_avg, color = 'cyan', marker ='*', label = 'Avg. Err. Gauss. Cons.')
plt.plot(frac_stragglers, opt_avg_2, color = 'red', marker ='s', label = 'Avg. Err. Base. Cons.')
plt.plot(frac_stragglers, new_lower_bound, color = 'Orange', marker ='x', label = 'Low. Bd.')




plt.xlabel(r'Straggler Fraction $(\frac{s}{n})$', fontsize = 15)
plt.ylabel('Approximation Error', fontsize = 15)
plt.legend(fontsize='large',loc='upper left')
plt.rc('text', usetex=True)
plt.grid(True)
#plt.title('Approximation Error vs Fraction of Workers Straggling', fontsize = 14)
plt.savefig('ISIT_2025_plot_5(40, 20, 3, 6 bipartite)_v_2.pdf')





