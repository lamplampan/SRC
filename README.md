SRC
===

Sparse Representation-based Classification, a classifier algorithm in Pattern Recognition area

+++

Author  : Denglong Pan
          pandenglong@gmail.com

+++

What is SRC ? 

Refer to https://github.com/lamplampan/SRC/wiki/Home-for-SRC#what-is-src

# What is SRC

SRC, Sparse Representation-based Classification, is an algorithm in Pattern Recognition area.

### Sparse linear combination

Training sample vector

<img src="http://latex.codecogs.com/gif.latex?A_{i}&space;=&space;[&space;v_{i,1}&space;,&space;v_{i,2}&space;,&space;...&space;,&space;v_{i,n_{i}}&space;]\in&space;IR^{m\times&space;n_{i}}" title="A_{i} = [ v_{i,1} , v_{i,2} , ... , v_{i,n_{i}} ]\in IR^{m\times n_{i}}" />

Test sample y be projected in the relevant linear space of its class.

<img src="http://latex.codecogs.com/gif.latex?y&space;=&space;a&space;_{i,1}v&space;_{i,1}&space;&plus;&space;a&space;_{i,2}v&space;_{i,2}&space;&plus;&space;...&space;&plus;&space;a&space;_{i,ni}v&space;_{i,ni}" title="y = a _{i,1}v _{i,1} + a _{i,2}v _{i,2} + ... + a _{i,ni}v _{i,ni}" />

Training sample set

<img src="http://latex.codecogs.com/gif.latex?A&space;=&space;[&space;A_{1},&space;A_{2},...,&space;A_{k}&space;]&space;=&space;[&space;v_{1,1},&space;v_{1,2},...,&space;v_{k,n_{k}}&space;]" title="A = [ A_{1}, A_{2},..., A_{k} ] = [ v_{1,1}, v_{1,2},..., v_{k,n_{k}} ]" />

Training sample y should be 

<img src="http://latex.codecogs.com/gif.latex?y&space;=&space;Ax_{0}&space;\in&space;IR^{m}" title="y = Ax_{0} \in IR^{m}" />

In which the x。should be 

<img src="http://latex.codecogs.com/gif.latex?x_{0}&space;=&space;[0,...,0,a_{i,1},&space;a_{i,2},...,a_{i,n_{i}},&space;0,...,0]^{T}&space;\in&space;IR^{n}" title="x_{0} = [0,...,0,a_{i,1}, a_{i,2},...,a_{i,n_{i}}, 0,...,0]^{T} \in IR^{n}" />

It is not only one solution for y = Ax  since it is always underdetermined.

We could determine the solution by calculate the minimum L2 norm.

(L2)   <img src="http://latex.codecogs.com/gif.latex?\hat{x}_{2}&space;=&space;arg[&space;min&space;||x||_{2}&space;]" title="\hat{x}_{2} = arg[ min ||x||_{2} ]" />       In which the Ax = y.


The resolved x2 is dense in general. It is hard to be classified.

In stead, we will looking for the sparsest resolution for the following formula, in which the Ax = y. The procedure to solve is a NP problem.

(L0)   <img src="http://latex.codecogs.com/gif.latex?\hat{x}_{0}&space;=&space;arg[&space;min&space;||x||_{0}&space;]" title="\hat{x}_{0} = arg[ min ||x||_{0} ]" />

We could find the approximate solution of L1 norm instead of L0 norm.

(L1)   <img src="http://latex.codecogs.com/gif.latex?\hat{x}_{1}&space;=&space;arg[&space;min&space;||x||_{1}&space;]" title="\hat{x}_{1} = arg[ min ||x||_{1} ]" />        in which the Ax = y.

L1 norm can be resolved by standard linear programming tool for polynomial. It will be more efficient if the resolution is sparse.

There is noise in the real data,

<img src="http://latex.codecogs.com/gif.latex?y&space;=&space;Ax_{0}&space;&plus;&space;z" title="y = Ax_{0} + z" /> in which <img src="http://latex.codecogs.com/gif.latex?z\in&space;IR^{m}" title="z\in IR^{m}" /> is a noise. 

So , x。can be resolved by approximate solution of L1 norm.

<img src="http://latex.codecogs.com/gif.latex?(L^{_{s}^{l}})&space;\hat{x}_{1}&space;=&space;arg&space;[min||x||_{1}]" title="(L^{_{s}^{l}}) \hat{x}_{1} = arg [min||x||_{1}]" />    , in which <img src="http://latex.codecogs.com/gif.latex?||&space;Ax&space;-&space;y&space;||_{2}&space;\leqslant&space;\varepsilon" title="|| Ax - y ||_{2} \leqslant \varepsilon" />

### Sparse Representation

Use <img src="http://latex.codecogs.com/gif.latex?&space;\delta&space;_{i}" title="\delta _{i}" /> as the eigenfunction to filter the corresponding coefficient of class i. It is a new vector in which the non-zero values are related to class i. We can classify y by the minimum residual between y and <img src="http://latex.codecogs.com/gif.latex?&space;\hat{y}_{i}" title="\hat{y}_{i}" /> 

<img src="http://latex.codecogs.com/gif.latex?&space;min_{i}&space;:&space;r_{i}(y)&space;=&space;||y&space;-&space;A\delta&space;_{i}&space;(\hat{x_{i}})||_{2}" title="min_{i} : r_{i}(y) = ||y - A\delta _{i} (\hat{x_{i}})||_{2}" />

### SRC Algorithm

1 Input : Training sample matrix 

<img src="http://latex.codecogs.com/gif.latex?&space;A&space;=&space;[&space;A_{1},&space;A_{2},&space;...&space;,&space;A_{k}]&space;\in&space;IR^{m&space;\times&space;n}" title="A = [ A_{1}, A_{2}, ... , A_{k}] \in IR^{m \times n}" />

for k kinds of classes, test samples <img src="http://latex.codecogs.com/gif.latex?&space;y&space;\in&space;IR^{m&space;\times&space;n}" title="y \in IR^{m \times n}" /> , Fault tolerance coefficient  <img src="http://latex.codecogs.com/gif.latex?&space;\varepsilon&space;\geqslant&space;0" title="\varepsilon \geqslant 0" />

2 Normalize the columns of A , to make L2 norm unit. 

3 Minimize L1 : 

<img src="http://latex.codecogs.com/gif.latex?&space;\hat{x}_{1}&space;=&space;arg&space;[min||x||_{1}]" title="\hat{x}_{1} = arg [min||x||_{1}]" />   ,   in which <img src="http://latex.codecogs.com/gif.latex?&space;||Ax&space;-&space;y&space;||_{2}&space;\leq&space;\varepsilon" title="||Ax - y ||_{2} \leq \varepsilon" />

4 Calculate the residual. 

<img src="http://latex.codecogs.com/gif.latex?&space;r_{i}(y)&space;=&space;||&space;y&space;-&space;A\delta&space;_{i}&space;(\hat{x}_{1})_{2}" title="r_{i}(y) = || y - A\delta _{i} (\hat{x}_{1})_{2}" />

5 Output : Make choice among <img src="http://latex.codecogs.com/gif.latex?&space;(y)&space;=&space;arg&space;[min_{i}r_{i}(y)]" title="(y) = arg [min_{i}r_{i}(y)]" /> 

### Test result on ORL face lib.

There are 40 classes in ORL face lib. Choose 5 as training samples in each class. Choose the left 5 as the test samples in each class.

Firstly use the PCA as feature extraction. Project the training samples on principal vectors. Then classify the test samples by SRC classifier. 

The recognition rate is 96% when the PCA=80 .


+++

How to run the algorithm?

Refer to https://github.com/lamplampan/SRC/wiki/Home-for-SRC#how-to-run

# How to run

1 Config the two ORL face lib path in file orl_src.m . Default path is E:\ORL_face\orlnumtotal\

2 Run orl_src.m 

