

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import itertools



A = np.load('BIBD91.npy')






n_1 = A.shape[0]
n_2 = A.shape[1]
d_1 = (A@A.T)[0,0]
d_2 = (A.T@A)[0,0]


RR = A.T@A
lmd = RR[0,1]



print(n_2,n_1,d_2,d_1, lmd)



m = 2

F = np.kron(np.eye(m), np.ones((n_1,1)))

E = np.vstack((A,A))



p = 15
n_w = E.shape[1]
n_p = n_1
s_list= np.linspace(0, (.25*(n_w)), p + 1, dtype=int)[1:]

frac_stragglers = s_list/n_w

d_choices = 100
itr = 1000

exp_opt_error = np.zeros((len(s_list)))
exp_fixed_diag = np.zeros((len(s_list), d_choices))



for i in range(len(s_list)):
    l = 0
    for k in range(d_choices):
            

            np.random.seed(k)
            d_v_1 = np.random.choice([1, -1], size= (n_2), p=[0.5, 0.5])
            D1 = np.diag(d_v_1)
            d_v_2 = np.random.choice([1, -1], size= (n_2), p=[0.5, 0.5])
            D2 = np.diag(d_v_2)
            E1 = np.vstack((A@D1, A@D2))

            c = 0
            for j in range(itr):
                    #print(i,j)
                    np.random.seed(j+k)
                    random_non_stragglers = np.random.choice(np.arange(0, n_w), int(n_w-s_list[i]), replace=False)
                    random_non_stragglers = np.sort(random_non_stragglers)
                    E_F_1 = E1[:, random_non_stragglers]
      
                    opt_err = 0
                    opt_err_1 = 0
                    exp_opt_err_1 = 0

                    for t in range(F.shape[1]):
    
                              a = np.reshape(a, (m*n_p,1))
                              #print(a)


                              

       
                             
                              x, residuals, rank, s = np.linalg.lstsq(E_F_1, a, rcond=None)
                              opt_err_1 = opt_err_1 + np.linalg.norm(E_F_1@x-a,2)**2
                              
        

                    #print(opt_err_1.item())
                    #opt_err_array[i,j] = np.sqrt((np.real(opt_err.item())))
                    c = c +  ((np.real(opt_err_1.item())))
                  

            #print(c)
            exp_fixed_diag[i,k] = c/itr
            l = l+c
            


    exp_opt_error[i] = l/(d_choices*itr)
    

new_lower_bound = np.zeros((len(s_list)))
lb = np.zeros((m))
for i in range(len(s_list)):
  for j in range(m):
      lb[j] = np.floor((s_list[i] + m - j -1)/d_2) * (j+1)
  new_lower_bound[i] = np.max(lb)

opt_err_array = np.zeros((len(s_list), itr))
opt_max = np.zeros(len(s_list))
opt_min = np.zeros(len(s_list))
opt_avg = np.zeros(len(s_list))

for i in range(len(s_list)):
    for j in range(itr):
            #print(i,j)
            np.random.seed(78+j)
            
            random_stragglers = np.random.choice(np.arange(0, int(n_w)), int(s_list[i]), replace=False)
            random_non_stragglers = np.setdiff1d(np.arange(0, n_w),random_stragglers)
            random_non_stragglers = np.sort(random_non_stragglers)
            E_F = E[:, random_non_stragglers]
            
            
            opt_err = 0
            
            for t in range(m):

                      a = F[:,t]
                      a = np.reshape(a, (m*n_p,1))

                      #print(a)
                      # print(a.conj().T@a-a.conj().T@E_F@ np.linalg.inv(E_F.conj().T@E_F).conj().T@E_F.conj().T@a)



                      
                      x, residuals, rank, s = np.linalg.lstsq(E_F, a, rcond=None)
                      opt_err = opt_err + np.linalg.norm(E_F@x-a,2)**2
                      
            
            opt_err_array[i,j] = (np.real(opt_err))
            



    opt_max[i] = np.max(opt_err_array[i,:])
    opt_min[i] = np.min(opt_err_array[i, :])
    opt_avg[i] = np.mean(opt_err_array[i, :])

   

exp_bound_bernoulli = (m*n_p - ((m*d_2**2*(n_w-s_list))/((m*d_2 - lmd) + lmd*(n_w-s_list))))


opt_trivial = (m*n_p - ((d_2**2*(n_w-s_list))/((d_2 - lmd) + lmd*(n_w-s_list))))



plt.plot(frac_stragglers,exp_bound_bernoulli, color = 'green', marker ='o', label = r'Upp. Bd.')
plt.plot(frac_stragglers, exp_opt_error, color = 'blue', marker ='^', label = 'Emp. Avg. Err.')
plt.plot(frac_stragglers, opt_avg, color = 'red', marker ='s', label = 'Err. Base. Cons.')
plt.plot(frac_stragglers, new_lower_bound, color = 'Orange', marker ='x', label = 'Low. Bd.')



plt.xlabel(r'Straggler Fraction $(\frac{s}{n})$', fontsize = 15)
plt.ylabel('Approximation Error', fontsize = 15)
plt.legend(fontsize='large',loc='upper left')
plt.rc('text', usetex=True)
plt.grid(True)
#plt.title('Approximation Error vs Fraction of Workers Straggling', fontsize = 15)
plt.savefig('ISIT_2025_plot_1(91,91,10,10)_BIBD_v_2.pdf')



