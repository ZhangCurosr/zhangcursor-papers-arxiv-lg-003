# Variational Functional Neural Networks (vfNN)

Chinmay Katke<sup>1,2</sup> and C. Nadir Kaplan<sup>1−4∗</sup>

<sup>1</sup>Department of Physics, Virginia Polytechnic Institute and State University, Blacksburg, VA 24061, USA.

<sup>2</sup>Center for Soft Matter and Biological Physics, Virginia Polytechnic

Institute and State University, Blacksburg, VA 24061, USA.

<sup>3</sup>Department of Mechanical Engineering, Virginia Polytechnic Institute and State University, Blacksburg, VA 24061, USA.

<sup>4</sup>Center for the Mathematics of Biosystems, Virginia Polytechnic Institute and State University, Blacksburg, VA 24061, USA. (Dated: November 3, 2026)

We present a variational functional neural network (vfNN) algorithm that optimizes physically motivated continuum functionals in integral form (e.g., Lagrangian, Hamiltonian, or free energy) to find the states of a system. That way, vfNN bypasses solving the equations of motion (a.k.a. the Euler-Lagrange equations), which are in the form of diferential equations. To this end, our approach is fundamentally diferent from conventional physics-informed neural networks (PINN) that solve diferential equations by minimizing a phenomenological loss function involving higher order derivatives and ad hoc weight factors. In contrast, vfNN merely evaluates lower order derivatives of the generating functional and replaces the weight factors with rigorous physical scales. This also fundamentally changes the loss landscape, and the desired solutions may become unstable. For this emergent nonconvex optimization problem, we develop a quasi-Newton optimization algorithm with simulated annealing that converges to the solutions of the corresponding Euler-Lagrange equations of the functional regardless of their stability properties. Our simulations using vfNN give accurate results for benchmark systems governed by linear or nonlinear ordinary diferential equations.

## I. INTRODUCTION

The equations of motion of a wide range of dissipative or energy conserving physical systems can be derived from a Lagrangian, Hamiltonian, or Rayleighian functional that involves conservative or nonconservative forces. The analytical extremization yields Euler-Lagrange equations (ELE), which inherently employ gradient information to solve for a smooth function, such as the position of a body in time, or an order parameter and a nonuniform vector field of a many-body system.

Neural networks as universal function approximators and reliable gradient algorithms have thus become appealing for automating the solution of physically motivated diferential equations [1, 2]. To this end, physicsinformed neural networks (PINN) have become a popular tool to solve for the state of a system by minimizing a phenomenological loss function constructed from a set of coupled diferential equations [3, 4]. This loss function is the L<sup>2</sup> norm of the set, and the derivative terms in the loss function are calculated using automatic diferentiation (AD) [2]. The procedure is mesh-free in the sense that the accuracy of the solution does not depend on the coarseness of the underlying mesh. This gives PINN a significant advantage over conventional diferential equation solvers in problems involving irregular domains and large deformations [5, 6]. Consequently, PINN found wide applicability in biomedicine, fluid and solid mechanics, materials science, geophysics, dynamical systems and control, and chemical engineering [7–19].

Despite the advantages of NN implementations in physical modeling, using AD to compute higher order derivatives, which lie at the heart of the mathematical description of physical systems, is expensive due to extensive mapping of all lower derivatives. Even if the diferential equations are cast to a coupled system of first-order equations, the corresponding optimization procedure using PINN involves construction of a single loss function with ad hoc weight factors multiplying each equation. These weight factors are estimated manually or algorithmically, increasing the complexity of the problem.

These shortcomings can be circumvented by turning to the variational formulation of the problem at hand, which naturally has lower order derivatives. A variational PINN framework that utilizes weak formulation has previously been implemented in the Petrov-Galerkin framework (VPINN), where a deep neural network (DNN) acts as a trial function, and the test functions are represented by prescribed functions or a second neural network [20– 24]. These formulations are similar to the finite element method (FEM), and the accuracy depends on the number of quadrature points in the domain and the choice of test functions. Another NN approach – much closer in spirit to the physically motivated variation as with Lagrangian and Hamiltonian mechanics – was explored in the context of the Poisson’s equation as an elliptic partial diferential equation that was numerically solved by directly minimizing a convex functional [25–33].

Here, we generalize this latter approach by developing a variational functional neural network (vfNN) algorithm to directly optimize a broader class of physically moti vated functionals that have nonconvex loss landscapes. This approach bypasses solving ELE, thus significantly difering from PINN, as well as from VPINN in that the variational form is not the weak form. We demonstrate that vfNN can successfully extremize a functional to find the solution(s) even when the loss landscape is not guaranteed to be locally or globally convex. We implement vfNN to a range of benchmark problems with a variational formulation, whose ELE would correspond to a single (linear or nonlinear) ordinary diferential equation (ODE) or a coupled set thereof.

This paper is organized as follows: in Sec. II, we present the details of the vfNN algorithm. In particular, we characterize the nonconvexity of the loss landscape to assess the stability of the solutions of ELE, and then develop a quasi-Newton optimization procedure that identifies the solutions regardless of their stability. In Sec. III, we utilize vfNN to extremize the functionals of diferent physical systems whose ELEs constitute a linear ODE, a nonlinear ODE for the minimal surface of revolution, steady-state Sine-Gordon equation, Poisson-Boltzmann equation, and a set of coupled ODEs for the packing of hard rods in a liquid crystalline colloidal monolayer. We provide a discussion and concluding remarks in Sec. IV.

## II. METHODS

As a basic variational problem, we seek a function $\hat { \mathbf { u } } ( \mathbf { x } )$ that optimizes the integral in a domain $\Omega \subset \mathbb { R } ^ { d }$

$$
\mathcal { I } = \int _ { \Omega } F ( \mathbf { x } , \mathbf { u } ( \mathbf { x } ) , \mathbf { u } ^ { \prime } ( \mathbf { x } ) ) d \mathbf { x } , \quad \mathbf { x } = ( x _ { 1 } , x _ { 2 } , . . . , x _ { d } ) \in \Omega ,\tag{1}
$$

$$
{ \bf u } = ( u _ { 1 } , u _ { 2 } , . . . , u _ { n } ) , \qquad u _ { k j } ^ { \prime } \equiv \frac { \partial u _ { k } } { \partial x _ { j } } .
$$

Here $F$ is a scalar function of unknowns $\left( u _ { 1 } , u _ { 2 } , . . , u _ { n } \right)$ and their gradients $( \mathbf { u _ { 1 } ^ { \prime } } , \mathbf { u _ { 2 } ^ { \prime } } , . . , \mathbf { u _ { n } ^ { \prime } } )$ . The optimization is subject to the following Dirichlet (D) and Neumann (Nm) boundary conditions:

$$
\begin{array} { r l } & { \mathbf { u } ( \mathbf { x } ) = \mathbf { u } _ { D } , \qquad \mathbf { x } \in \partial \Omega _ { D } , } \\ & { \mathbf { u } ^ { \prime } ( \mathbf { x } ) = \mathbf { u } _ { N m } , \qquad \mathbf { x } \in \partial \Omega _ { N m } . } \end{array}\tag{2}
$$

The Dirichlet and Neumann boundary conditions hold on diferent parts of the domain boundary $\partial \Omega _ { D }$ and $\partial \Omega _ { N m }$ , respectively, where known functions $\mathbf { u } _ { D }$ and $\mathbf { u } _ { N m }$ are imposed.

By considering first-order variation of $\operatorname { E q . }$ 1 with respect to u , we arrive at the corresponding Euler-Lagrange equations (ELE):

$$
\frac { \partial F } { \partial u _ { k } } = \frac { d } { d x _ { j } } \frac { \partial F } { \partial u _ { k j } ^ { \prime } } .\tag{3}
$$

The solution $\hat { \mathbf { u } } ( \mathbf { x } )$ to the ELE is defined as the first-order critical point in variational calculus [34].

To solve Eq. 1, we represent $\mathbf { u } ( \mathbf { x } )$ using a neural network (NN) as a universal function approximator, which reduces the original infinite-dimensional optimization problem to finite dimensions [1]. A point $\mathbf { x } _ { i } \in \Omega$ is an input to the neural network, and the corresponding output is $\mathbf { u _ { N } } ( \mathbf { x } _ { i } )$ . We denote the set of such points in the domain by $\{ { \bf { x } } _ { i } \}$ (collocation points) and the set of the outputs from the neural network by $\{ \mathbf { u } _ { \mathbf { N } } ( \mathbf { x } _ { i } ) \}$

The network topology that we use for vfNN is depicted in Fig. 1. We construct the NN as follows: let $\mathbf { o } ^ { 0 } \equiv \mathbf { x } _ { i }$ denote the input to the network, and let m be the number of layers. For each layer $l = 1 , \ldots , m$ with the number of neurons $n _ { l }$ , let $\mathbf { W } ^ { l } \in \mathbb { R } ^ { n _ { l } \times n _ { l - 1 } }$ be the weight matrix, and $\mathbf { b } ^ { l } \in \mathbb { R } ^ { n _ { l } }$ be the bias vector that enables nonzero output of the neural network for zero input $( \mathbf { x } = 0 )$ The output of the $( l - 1 ) ^ { \mathrm { t h } }$ layer is denoted by $\mathbf { o } ^ { l - 1 }$ . Then,

$$
\mathbf { a } ^ { l } = \mathbf { W } ^ { l } \cdot \mathbf { o } ^ { l - 1 } + \mathbf { b } ^ { l } , \quad l = 1 , 2 , \ldots , m ,\tag{4}
$$

$$
\mathbf { o } ^ { l } = \sigma ^ { l } \left( \mathbf { a } ^ { l } \right) , \quad l = 1 , 2 , \ldots , m .\tag{5}
$$

The activation functions are defined as

$$
\sigma ^ { l } ( x ) = \operatorname { t a n h } ( x ) , \quad l = 1 , 2 , \ldots , m - 1 ,
$$

$$
\sigma ^ { m } ( x ) = x , \quad \mathrm { ( i d e n t i t y ~ f u n c t i o n ) } .\tag{6}
$$

(7)

We choose the hyperbolic tangent function as an activation function for all the layers except for the last layer, which allows for a simple pass-through to give the output $o ^ { m } = \mathbf { u _ { N } } ( \mathbf { x _ { i } } )$ . For convenience, we denote the set of all parameters that are the entries of the matrices $\mathbf { W } ^ { l }$ and vectors $\mathbf { b } ^ { l }$ by a single vector $\mathbf { w } \in \mathbb { R } ^ { n _ { t } }$ , where $n _ { t }$ is the total number of NN parameters. We choose the most fundamental neural network, multilayer perceptron (MLP), as the underlying network structure. Furthermore, we set three hidden layers (length) with 20 neurons in each layer (width). These network dimensions are optimal for the validation problems considered in this work. Networks with number of layers less than three and a width less than 20 were not suficient to approximate the solutions. We found that MLP with bigger length and width higher increased computational cost without significantly improving the accuracy.

In the conventional formulation of PINN, the ELE is solved by minimizing the corresponding $\mathcal { L } ^ { 2 }$ norm $[ 3 ]$ :

$$
\mathcal { L } _ { P I N N } = \sum _ { i } \mu _ { i } \sum _ { j } \left( \frac { \partial F } { \partial u _ { i } } - \frac { d } { d x _ { j } } \frac { \partial F } { \partial _ { j } u _ { i } ^ { \prime } } \right) ^ { 2 } .\tag{8}
$$

Here, the sum over index i runs over the Euler-Lagrange equations, each corresponding to an unknown $u _ { i }$ describing the system, and $\mu _ { i }$ are the corresponding loss weight factors. The determination of these weight factors is not trivial and often involves trial and error or implementation of complex algorithms. In vfNN, to obtain the solution of Eq. 3, we optimize Eq. 1 where rigorous physical scales describing the system efectively replace the loss weight factors $\mu _ { i }$ in Eq. 8. To this end, we first approximate the integral in Eq. 1 by a right Riemann sum:

$$
\mathcal { T } \approx \frac { V } { N } { \sum _ { i } ^ { \vphantom { N } } } F ( { \bf x _ { i } } , { \bf u } ( { \bf x _ { i } } ) , { \bf u } ^ { \prime } ( { \bf x _ { i } } ) ) .\tag{9}
$$

Here $\begin{array} { r } { V = \int \cdots \int _ { V } d x _ { 1 } \dots d x _ { d } } \end{array}$ is the volume of the domain Ω , and $N$ is total number of collocation points. For a fixed set of collocation points $\{ \mathbf { x } _ { i } \} , ~ \mathcal { I }$ is a function of w , i.e., the variational problem in Eq. 1 can be solved by optimizing the NN weights and biases.

![](images/613c54ad22a099cf98f1cbbaaed24f55d6ee0879a799584b3f5742c70ed860cd.jpg)  
FIG. 1. Variational functional neural network (vfNN). We use a multilayer perceptron (MLP) deep neural network with an input x and output u that minimizes the variational loss $\mathcal { I }$ . The derivative ${ \bf { u } } ^ { \prime } ( { \bf { x } } )$ is computed using automatic diferentiation at each input point x . The Dirichlet boundary conditions are implemented by transforming the output of the network $\mathbf { u _ { N } }$ with known functions $\tilde { l } ( { \bf x } )$ and $g ( \mathbf { x } )$ . The Neumann boundary conditions are implemented by including a corresponding boundary term in the functional $\mathcal { I }$ . For the definitions of $\sigma , l .$ , W, o , see Eqs. 4, 5.

The most popular method to implement the boundary conditions $\mathrm { ( B C s ) }$ in PINN is to include a corresponding penalty term for the deviation from the desired BC in the overall loss function [3]. For the Dirichlet BCs stated in problem 1, the penalty terms are given by

$$
\mathcal { L } _ { B C } = \mu \sum _ { i } ( \mathbf { u _ { N } } ( \mathbf { x _ { i } } ) - \mathbf { u } _ { D } ) ^ { 2 } , \qquad \mathbf { x } _ { i } \in \Omega _ { D } ,\tag{10}
$$

where $\mu$ is again a penalty coeficient. Generally, the optimization is initiated with a smaller value $\mu$ whose value is increased gradually until the BCs are met. Since this method adds an extra layer of complexity to the optimization process, in vfNN, we instead implement the Dirichlet BCs for the problem in Eq. 1 by directly constraining the output of the neural network [35]:

$$
\mathbf { u } ( \mathbf { x } ) = \tilde { l } ( \mathbf { x } ) \mathbf { u _ { N } } ( \mathbf { x } ) + g ( \mathbf { x } ) .\tag{11}
$$

The functions $\tilde { l } ( { \bf x } )$ and $g ( \mathbf { x } )$ are any smooth function over the domain $\mathbf { x } \in \Omega$ with following properties:

$$
\begin{array} { r l r } & { \tilde { l } ( \mathbf { x } _ { i } ) \neq 0 , } & { \mathbf { x } _ { i } \in \Omega , } \\ & { \tilde { l } ( \mathbf { x } _ { i } ) = 0 , } & { \mathbf { x } _ { i } \in \Omega _ { D } , } \\ & { g ( \mathbf { x } _ { i } ) = \mathbf { u } _ { D } , } & { \mathbf { x } _ { i } \in \Omega _ { D } . } \end{array}\tag{12}
$$

The simplest function one can construct to satisfy the above criteria are $\tilde { l } ( { \bf x } )$ quadratic in x and $g ( \mathbf { x } )$ linear in x. Hence, we have chosen $\tilde { l } ( { \bf x } )$ and $g ( \mathbf { x } )$ accordingly for each problem studied in this work.

Eq. 8 ensures that the solution is a local minimum in the multidimensional loss landscape, and the optimization algorithms such as the adaptive moment estimation (ADAM) and the limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm (L-BFGS) work very well to find the solution [36, 37]. In contrast, although the solution ˆu to Eq. 3 is the optimum solution to the problem defined by Eq. 1, it is not guaranteed that the optimum corresponds to a minimum (See Appendix A for the conditions of stability). In general, the solution ˆu could be located at a maximum, minimum, or a saddle point in the landscape of $\mathcal { I }$ ; this increases the complexity of optimization. To overcome this, Appendix B outlines a simple algorithm to determine the stability of a 1D functional whose ELE is a linear ODE.

To find the solution ˆu of the general variational problem in $\operatorname { E q . 9 }$ , we develop an optimization algorithm that converges to a first-order critical point based on Newton’s method of optimization. Newton’s method uses the first derivatives as well as the second derivatives or the Hessian matrix of the objective function to find the optimum (minimum, maximum or a saddle point), ofering faster convergence compared to other first-order methods such as the gradient descent algorithm. However, it can sufer from higher computational and memory costs associated with the second derivatives in a large parameter space, in our case of dimension $n _ { t }$ of the weight vector w. The quasi-Newton second-order methods like the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm approximate the Hessian of the objective function using first derivatives and the step size in the parameter space, thereby reducing the computational cost. The second-order optimization methods are typically combined with supplementary algorithms to accelerate convergence. While quasi-Newton methods determine the search direction in the parameter space, the supplementary “line search” algorithms determine the optimum step size in that direction. A variety of line search algorithms satisfy criteria called the Wolfe conditions that impose suficient decrease (Armijo) and the curvature conditions, ensuring convergence to a global minimum when combined with the BFGS algorithm [38].

The conventional BFGS method approximates the true Hessian matrix by using another matrix with a strictly positive eigenvalue spectrum. In the vicinity of a local minimum, the Hessian is always positive definite (i.e., has only positive eigenvalues), and hence BFGS works very well at locating minima. While this method is perfectly suited for the task of finding a minimum, it is inefective for finding a solution to variational problems when the solution lies at a saddle point or a maximum in the loss landscape. This is because around the saddle points, the true Hessian has both positive and negative eigenvalues.

Thus, we use $\mathrm { S y }$ mmetric Rank 1 (SR1) Hessian update with a line search method based on a Metropolis criterion [38, 39]. The advantage of using SR1 Hessian update is that the approximate Hessian is not constrained to represent only the positive eigenvalues, accurately approximating the true Hessian matrix when it is indefinite $( { \mathrm { i . e . } }$ , has positive, zero, and negative eigenvalues) in the vicinity of a saddle point or inflection point solution.

We implement the line search method based on a Metropolis criterion as follows: let $\mathbf { w } _ { k } = ( w _ { 1 } , . . . , w _ { n } )$ be a vector where the elements $w _ { i }$ denote the NN weights and biases, and the subscript k denotes the $k ^ { \mathrm { t h } }$ iteration. Let $\mathcal { I } ( \mathbf { w _ { k } } ) \equiv \mathcal { I } _ { k }$ be the variational loss function at the iteration $k , \ \nabla \mathcal { I } _ { k }$ be the gradient in the loss landscape calculated using backpropagation, and $B _ { k }$ is an n × n matrix that approximates the Hessian matrix of $\mathcal { T } _ { k }$ . The matrix $B _ { k }$ is evaluated iteratively during optimization. We define the vector $\mathbf { p } \in \mathbb { R } ^ { n _ { t } }$ as the direction of the next iteration step, $\mathrm { i . e . , }$ the line search direction. Then, the quadratic approximation $m _ { k } ( \mathbf { p } )$ of the variational loss about $\mathcal { T } _ { k }$ is given by

$$
m _ { k } ( { \bf p } ) \equiv \mathcal { I } ( { \bf w } _ { \bf k } + { \bf p } ) = \mathcal { I } _ { k } + \nabla \mathcal { I } _ { k } ^ { T } { \bf p } + \frac { 1 } { 2 } { \bf p } ^ { T } B _ { k } { \bf p } .\tag{13}
$$

By also expanding $\nabla m _ { k } ( \mathbf { p } )$ into a Taylor series, the optimum of the quadratic approximation in Eq. 13 is obtained by setting $\nabla m _ { k } ( { \mathbf p } ) = 0 ;$

$$
\mathbf { p } _ { k } = - B _ { k } ^ { - 1 } \nabla \mathcal { I } _ { k } + \mathcal { O } ( \mathbf { p } ^ { 2 } ) .\tag{14}
$$

$\operatorname { E q } .$ . 14 is used to calculate the next iteration,

$$
\mathbf { w } _ { k + 1 } = \mathbf { w } _ { k } + \alpha _ { k } \mathbf { p } _ { k } ,\tag{15}
$$

where $\alpha _ { k }$ is the step size.

Algorithm 1: Stochastic line search   
Input: $\mathbf { w } _ { k } , \nabla \mathcal { T } ( \mathbf { w } _ { k } ) , B _ { k } ^ { - 1 } , \gamma , \beta , \alpha _ { \mathrm { m i n } }$   
1 $\begin{array} { r } { \alpha  1 , \mathbf { p }  - B _ { k } ^ { - 1 } \nabla \mathcal { T } ( \mathbf { \bar { w } } _ { k } ) , i  0 ; } \end{array}$   
2 repeat   
3 $\phi ( \alpha )  \| \nabla \mathcal { I } ( \mathbf { w } _ { k } + \alpha \mathbf { p } _ { k } ) \| ;$   
4 if uniform(0, 1) ≤ exp(−β[Φ(α) − Φ(0)]) then   
5 return α;   
6 else   
7 α ← 0.9 α;   
8 $i \gets i + 1 ;$   
9 until $i = i _ { \mathrm { m a x } } ;$   
10 return $\alpha _ { \mathrm { { m i n } } } ;$

To find an appropriate step size $\alpha _ { k }$ , our line search method given in Algorithm 1 employs a Metropolis criterion. The resulting value of $\alpha _ { k }$ reduces the magnitude of the objective function gradient along the line search direction p towards $\lVert \nabla \mathcal { I } _ { k } \rVert \approx 0$ where the optimum of $\mathcal { I }$ is located. While typical line search algorithms may utilize a variety of deterministic criteria to find an optimal step size $\alpha _ { k }$ , such as the Wolfe conditions that impose the suficient decrease (Armijo) and the curvature conditions [38], our algorithm is stochastic and is thus much less sensitive to noise in a potentially rough and nonconvex loss landscape. At every iteration $k ,$ we start with the maximum allowed step size $\alpha \equiv \alpha _ { k } = 1$ and inverse “temperature” $\beta$ for a Newton update. The Metropolis criterion

$$
u n i f o r m ( 0 , 1 ) \leq e ^ { - \beta [ \Phi ( \alpha ) - \Phi ( 0 ) ] } , \quad \Phi ( \alpha ) \equiv \frac { \phi ( \alpha ) } { \phi ( 0 ) ^ { - \gamma } } ,\tag{16}
$$

will always be true if $\phi ( \alpha ) = \lVert \nabla \mathcal { I } ( \mathbf { w } _ { k } + \alpha \mathbf { p } _ { k } ) \rVert$ is smaller than $\phi ( 0 )$ , and the step length $\alpha _ { k }$ will be accepted. Here, un $; f o r m ( 0 , 1 )$ corresponds to a random number between 0 and 1 , sampled from the uniform distribution at each iteration, which leads to occasional acceptance of a step length with $\phi ( \alpha ) > \phi ( 0 )$ . This helps navigate the highly nonconvex loss landscape of the neural network to find the optimum. We rescale the cost function $\phi ( \alpha )$ with $\phi ( 0 ) ^ { - \gamma }$ to ensure that the algorithm does not randomly escape the basin of the optimum; $\gamma > 0$ is a tuning parameter. While lower values of $\gamma$ increase the chances of escaping the basin of the optimum, higher values of γ result in very slow convergence. Thus, we choose $\gamma = 0 . 5$ If the Metropolis criterion is not met, then the step length α is reduced by a factor $( = 0 . 9$ here), and the process is repeated. The failure to satisfy the Metropolis condition for the maximum number of iterations would return a small constant step length $\alpha _ { m i n }$

Having determined the line search direction p<sub>k</sub> (Eq. 14) and the step size $\alpha _ { k }$ (Algorithm 1), the SR1 Hessian update that relates the approximate Hessian matrices $B _ { k }$ and $B _ { k + 1 }$ is given as [38]

$$
\begin{array} { l } { { \displaystyle B _ { k + 1 } ~ = ~ B _ { k } + \frac { ( { \bf y } _ { k } - B _ { k } { \bf s } _ { k } ) ( { \bf y } _ { k } - B _ { k } { \bf s } _ { k } ) ^ { T } } { ( { \bf y } _ { k } - B _ { k } { \bf s } _ { k } ) ^ { T } { \bf s } _ { k } } } } \\ { ~ \equiv ~ S R 1 ( B _ { k } , { \bf s } _ { k } , { \bf y } _ { k } ) , } \end{array}\tag{17}
$$

with

$$
\mathbf { s } _ { k } = \mathbf { w } _ { \mathbf { k } + 1 } - \mathbf { w } _ { \mathbf { k } } = \alpha _ { k } \mathbf { p } _ { k } , \quad \mathbf { y } _ { k } = \nabla \mathcal { J } _ { k + 1 } - \nabla \mathcal { J } _ { k } .\tag{18}
$$

For the SR1 update scheme, we have defined the function $S R 1 ( B _ { k } , \mathbf { s } _ { k } , \mathbf { y } _ { k } )$ . Eq. 17 can be obtained by considering the secant equation

$$
B _ { k + 1 } { \mathbf s } _ { \mathbf k } = { \mathbf y } _ { \mathbf k } .\tag{19}
$$

The only symmetric rank-1 update satisfying Eq. 19 and $B _ { k + 1 } = B _ { k } + c \mathbf { v } \mathbf { v } ^ { \top } \left( c \in \mathbb { R } , \mathbf { v } \in \mathbb { R } ^ { n } , n \in \mathbb { R } \right)$ is given by Eq. 17 [38]. We start with $B _ { 0 } = \mathbf { I } .$ where I is the identity matrix. The inversion of an n × n matrix $B _ { k }$ required to solve $\operatorname { E q } .$ . 14 is impractical when $n \gg 1$ . The elegance of the quasi-Newton methods is that the inverse $\bar { B } _ { k } ^ { - 1 }$ can also be computed iteratively [38]:

$$
B _ { k + 1 } ^ { - 1 } = S R 1 ( B _ { k } ^ { - 1 } , { \bf s } _ { k } , { \bf y } _ { k } ) .\tag{20}
$$

The SR1 update becomes unstable when the vectors ${ \bf y } _ { k } -$ $B _ { k } { \bf s } _ { k }$ and $\mathbf { s } _ { k }$ are perpendicular to each other and their scalar product in the denominator in Eq. 17 vanishes. To avoid this numerical instability, we perform the update when the criterion below is met:

$$
\begin{array} { r } { \left| { \bf s } _ { k } ^ { T } ( { \bf y } _ { k } - B _ { k } { \bf s } _ { k } ) \right| \geq r \| { \bf s } _ { k } \| \| { \bf y } _ { k } - B _ { k } { \bf s } _ { k } \| , } \end{array}\tag{21}
$$

where $r \in ( 0 , 1 )$ is a very small number. If the above condition is not met, then $B _ { k + 1 } = B _ { k } \ [ 3 8 ]$

The procedure for the SR1 Hessian update with quenching is given in Algorithm 2, which employs simulated annealing with the inverse temperature $\beta$ varied through the optimization process [40, 41]. We start with low values of $\beta$ and quench the system by increasing β. The quenching rate $\tau$ is an additional tuning parameter. A lower quenching rate increases the likelihood of finding the optimum at a cost of higher number of iterations. The numerical values used for the parameters $T _ { 0 } , T _ { f } , \tau , \delta$ in Algorithm 2 can be found in Ref. [42].

Algorithm 2: SR1 Hessian update with   
simulated annealing   
Input: w: weights, $\mathcal { I } ( \mathbf { w } ) \colon$ loss functional, $\nabla { \mathcal { I } } ( \mathbf { w } ) { \mathrm { : } }$   
gradient, δ: tolerance, τ : quenching rate, $T _ { 0 } { \mathrm { : } }$   
initial temperature, $T _ { f } \colon$ final temperature   
1 $B ^ { - 1 } \gets { \bf I } , i \gets 0 ;$   
2 repeat   
1   
3 $\beta ( i )  \frac { } { ( T _ { 0 } - T _ { f } ) \ : \exp ( - \tau i ) + T _ { f } } ;$   
4 $\alpha \gets .$ Line search $\iota ( \beta ) ;$   
5 $\mathbf { w } _ { i + 1 }  \mathbf { w } _ { i } + \alpha B ^ { - 1 } \nabla \mathcal { I } ( \mathbf { w } _ { i } ) ;$   
6 $\mathbf { s } _ { i } \gets \mathbf { w } _ { i + 1 } - \mathbf { w } _ { i } , \quad \mathbf { y } _ { i } \gets \nabla \mathcal { I } ( \mathbf { w } _ { i + 1 } ) - \nabla \mathcal { I } ( \mathbf { w } _ { i } ) ;$   
7 $B ^ { - 1 } \gets \mathrm { S R 1 } ( B ^ { - 1 } , \mathbf { s } _ { i } , \mathbf { y } _ { i } ) ;$   
8 $i \gets i + 1 ;$   
9 until $\| \nabla { \mathcal { I } } \| < \delta ;$   
10 return $\mathbf { w } _ { i + 1 }$

## III. RESULTS

By using our vfNN algorithm where the vector of NN parameters ${ \bf w } _ { k }$ is optimized through Algorithm 2 over k iterations, we solve several representative linear and nonlinear ordinary diferential equations (ODEs) by using their variational formulation. In Figs. 3 and 4, we compare our results from vfNN with FEM results that we generated by using FEniCS 2019.1.0 [43].

We first consider the variational formulation of a linear second order ODE

$$
\mathcal { I } = \int _ { a } ^ { b } F ( x , u ( x ) , u ^ { \prime } ( x ) ) d x , \qquad ^ { F = u ^ { \prime } ( x ) ^ { 2 } + f ( u ) u , }\tag{22}
$$

whose ELE is

$$
\frac { d ^ { 2 } u } { d x ^ { 2 } } = \frac { 1 } { 2 } \left( \frac { \partial f } { \partial u } u + f \right) .\tag{23}
$$

Here, $f ( u )$ represents a known function up to linear order in u . Table I lists the Dirichlet boundary conditions that we implement at both endpoints of the respective domains for diferent choices of $f ( u )$ . The results for the choices of $f ( u )$ listed in Table I are depicted in Fig. 2a– e. Unlike conventional PINN where the loss function is an $\mathcal { L } ^ { 2 }$ norm $\left( \mathrm { E q . ~ } 8 \right)$ , the loss function of vfNN may converge to a nonzero value. One such example is $f ( u ) = u$ and Fig. 2f shows the convergence of the corresponding variational loss $\mathcal { I }$ as a function of iteration steps.

We also compare the convergence rates of vfNN and PINN, each subject to the loss optimization through our SR1 algorithm and the conventional ADAM algorithm [36]. For each of the network-algorithm pair, Table II presents the number of iterations necessary to achieve accurate solutions. The iterations are averaged over 30 independent initializations due to the randomness associated with our line search algorithm. Except for the case of $f ( u ) = ( 3 \pi / 2 ) ^ { 2 } u$ , vfNN along with SR1 algorithm for loss optimization requires least number of iterations for convergence. This computational advantage can be attributed to two factors; the reduced computational cost due to lower order derivatives in the loss function $\mathcal { I }$ and the superlinear convergence rate of the quasi-Newton SR1 optimization algorithm. Note that, for $f ( u ) = - ( 3 \pi / 2 ) \bar { ^ { 2 } u }$ , the use of ADAM optimizer on vfNN results in a divergence of the loss. In this case, the sole solution to the ELE resides at an unstable saddle point within the loss landscape, and the ADAM optimizer, which seeks the minimum of the loss function, is unable to locate the solution. Here, the first two suficiency conditions for a stable solution given in Appendix A are satisfied since $\hat { F } _ { u ^ { \prime } , u ^ { \prime } } = 1$ . Thus, it must be the third condition that is violated, in other words, the presence of the conjugate points makes the solution unstable. We find the sole conjugate point $x _ { c } = 0 . 6 6 6 6 8$ in the domain using our algorithm 3 (Appendix B) in 14 iterations which is accurate up to $4 ^ { \mathrm { t h } }$ decimal. This algorithm also finds multiple conjugate points, {0.66668, 1.33334} , in the extended domain $x \in [ 0 , 1 . 8 ]$ . Thus, we employ the SR1 optimizer to accurately identify the corresponding unstable solution in the domain $x \in [ 0 , 1 ]$ without the need to identify the conjugate points.

TABLE I. Functions f(u) and corresponding boundary conditions for Eqs. 22, 23. The results are shown in Fig. 2.  
f(u) Boundary conditions   
0 $\overline { { u ( - 1 . 0 ) = - 0 . 0 6 , } } \quad u ( 0 ) = 0 .$   
0.25 $u ( - 1 ) = 0 . 0 6 , \quad u ( 1 . 0 ) = 0 . 0 6$   
u $u ( - 1 ) = e ^ { - 1 } , \quad u ( 2 ) = e ^ { 2 } ,$   
$\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { - \left( \frac { \pi } { 2 } \right) ^ { 2 } u } \end{array}$ $u ( 0 ) = 1 , \quad u ( 1 ) = 0 .$   
$- \left( { \frac { 3 \pi } { 2 } } \right) ^ { 2 } u$ $u ( 0 ) = 1 , \quad u ( 1 ) = 0 .$

We next consider the functionals of a single dependent variable and its derivative whose ELEs are a single nonlinear ODE. First, we consider the problem of finding the minimal surface of revolution formed by the rotation of a two-dimensional curve around the x-axis. Since the desired function must always lie above the x-axis, such that it has no simple root, we impose the constraint $u ( x ) = [ \tilde { l } ( x ) u _ { N } ( x ) + \bar { g ( x ) } ] ^ { 2 }$ , where $u _ { N } ( x )$ is the output of the neural network and does not directly represent the dependent variable (see Fig. 1). The variational formulation of this problem is

$$
\mathcal { T } = \int _ { a } ^ { b } u \sqrt { 1 + u ^ { \prime } ( x ) ^ { 2 } } d x , \quad u ( x ) \geq 0 .\tag{24}
$$

We choose a domain $x \in [ 0 , 1 ]$ with the Dirichlet boundary conditions $u ( 0 ) = u ( \bar { 1 } ) \stackrel { \cdot } { = } 1$ . The special solution of

X  
![](images/ffdb0bcd2122ef3922a6d4f7fe5848caa463494252b69a60e9f9a34fa1356244.jpg)

![](images/3c5ad997c70cf27b2307bcefe00e80e3e35061c0b811216d88485b1a5314520f.jpg)

![](images/e1b3d19ab936e0384c0b85dd55668c828f520bd82f64c4824e08fe13c13285dc.jpg)

![](images/79ac362ec2d9b0a829b9486028cb7844f98dd0b6955ade3443899392c8eb2b91.jpg)

![](images/ed50d801e10a420dc7f8181992d6f99bb7d12cffb2f39bfb174b4be1709ffcb4.jpg)

![](images/11e93d509c669d5c16e8231c303504395d736cf79667afbdc6f16ac61dfa1260.jpg)

FIG. 2. Linear ODEs using vfNN. The function $f ( u )$ in $\operatorname { E q } .$ . 23 is chosen as (a) $f ( u ) = 0 , ( { \bf b } ) \ f ( u ) = 0 . 2 5 , ( { \bf c } ) \ f ( u ) =$ u , (d) $f ( u ) = - ( \pi / 2 ) ^ { 2 } u$ , and $( \mathbf { e } ) ~ f ( u ) = - ( 3 \pi / 2 ) ^ { 2 } u ^ { * }$ . (f) shows the variational loss J for $f ( u ) = u$ in (c) as a function of the iterations of our SR1 update (Algorithm 2). The dashed red lines show the solutions obtained using vfNN, and the solid blue lines indicate the solutions obtained using PINN. ∗ The domain in (e) contains a conjugate point that makes the loss landscape nonconvex; see Appendices A, B.
<table><tr><td rowspan=2 colspan=1>f(u)</td><td rowspan=1 colspan=2>vfNN</td><td rowspan=1 colspan=2>PINN</td></tr><tr><td rowspan=1 colspan=2>SR1 ADAM</td><td rowspan=1 colspan=2>SR1 ADAM</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Number of iterations</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>603</td><td rowspan=1 colspan=1>120</td><td rowspan=1 colspan=1>442</td></tr><tr><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>262</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>832</td></tr><tr><td rowspan=1 colspan=1>u</td><td rowspan=1 colspan=1>456</td><td rowspan=1 colspan=1>854</td><td rowspan=1 colspan=1>761</td><td rowspan=1 colspan=1>499</td></tr><tr><td rowspan=1 colspan=1> $\overline { { - \left( \frac { \pi } { 2 } \right) ^ { 2 } u } }$ </td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>229</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>413</td></tr><tr><td rowspan=1 colspan=1> $\left[ - \left( { \frac { 3 \pi } { 2 } } \right) ^ { 2 } u \right.$ </td><td rowspan=1 colspan=1>1793</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>1396</td><td rowspan=1 colspan=1>1985</td></tr></table>

TABLE II. Convergence rates of vfNN vs. PINN. By specifying diferent $f ( u )$ for the problem given in Eq. 22, we optimize PINN and vfNN via two algorithms: the first-order optimization algorithm ADAM [36] and the second-order optimization algorithm SR1 (Sec. II). For $f ( u ) = - ( 3 \pi / 2 ) ^ { 2 } u$ , the use of ADAM optimizer with vfNN leads to the divergence of the loss function since the domain contains a conjugate point that makes the loss landscape nonconvex; see Appendices A, B. For all cases, a deep neural network with three layers and 20 neurons per layer is used. The number of iterations are averaged over 30 independent initializations.

the problem can be found analytically as

$$
u ( x ) = \frac { 1 } { 2 \gamma } \mathrm { c o s h } \left[ \gamma ( 2 x - 1 ) \right] .\tag{25}
$$

The neck radius of the surface of revolution $u _ { m i n }$ satisfies $u _ { m i n } = 1 / 2 \gamma$ . There exist two solutions to this problem: while the solution corresponding to $\gamma = 0 . 5 8 9 3 8 8$ is a minimum, the solution corresponding to $\gamma = 2 . 1 2 6 8$ is a saddle point (Fig. 3a). Our vfNN algorithm successfully identifies both solutions by initializing the optimization from diferent initial guesses that are randomized by the initialization of the weight matrix $\mathbf { W } ^ { l }$ in Eq. 4.

The second nonlinear problem with a single dependent variable that we consider is the steady-state Sine-Gordon equation, which is also an integrable system and models the large-angle simple pendulum when the independent variable x represents time. The variational formulation and the corresponding ELE of the problem is as follows:

![](images/4ec67534afa89d1fb916346f58ba0db1387b792e4861948db6fc8041c62bfb85.jpg)

![](images/84bcb1cbc8562322690f258382b9b2f4009d39d330255fc0e07400ce778abba5.jpg)  
(c)

![](images/58b22a387f25b83e07176636b6464d036091b5e675513852721ca2bc56961c1b.jpg)  
FIG. 3. Nonlinear ODEs using vfNN. Solutions to the functionals whose ELE are (a) the minimal surface of revolution (Eqs. 24, 25), (b) steady-state Sine-Gordon equation (Eq. 26), and (c) the Poisson-Boltzmann equation (Eqs. 27– 29). The dashed red lines indicate the solutions using vfNN, and all other colors correspond to diferent nonunique solutions (a) from the exact solution (Eq. 25) and (b), (c) found by using FEM. For all panels, a deep neural network with three layers and 20 neurons at each layer is used.

$$
{ \mathcal { I } } = \int _ { a } ^ { b } \left[ { \frac { 1 } { 2 } } u ^ { \prime } ( x ) ^ { 2 } + \omega ^ { 2 } \cos u \right] d x , \quad { \frac { d ^ { 2 } u } { d x ^ { 2 } } } = - \omega ^ { 2 } \sin u .\tag{26}
$$

In the domain $x \in [ 0 , 1 ]$ imposing Dirichlet boundary conditions at both boundaries, $u ( 0 ) = 0$ and $u ( 1 ) = 1$ for $\omega = 6$ , this problem has three distinct solutions: solution 1 and solution 2 are stable (corresponding to the equilibrium of the pendulum at $u \ : = \ : 0$ and $u \ : = \ : 2 \pi )$ ， and solution 3 is unstable (corresponding to $u = \pi )$ . By changing the initial guess, vfNN locates all three solutions regardless of their stability in agreement with the FEM profiles (Fig. 3b).

![](images/0bc1389844213535082eb0a8770a20a820e127f3d1774a4355f07cc7cfc76eae.jpg)  
(b)

![](images/67a3f993938db3fb3a0a0539f8beb1ebbff46e92dac47b5add9ae863c5ca6d54.jpg)  
FIG. 4. Nonlinear ODEs using vfNN (continued). Minimization of the elastic energy of a colloidal membrane composed of hard rods (Eqs. 30–34) [44]. (a) $\theta ( x )$ the tilt angle with respect to membrane normal. (b) $\psi ( x )$ is the scalar order parameter describing the orientation of the rods with respect to the membrane normal. The x axis is nondimensionalized by the twist penetration depth $\lambda _ { t }$ . The membrane interior is at $x = 0$ , whereas $x = L = 1 0$ is the position of the membrane edge. The dashed red lines indicate the solutions using vfNN, and the blue curves correspond to the solutions from FEM. Two separate deep neural networks (three layers with 20 neurons at each layer) for each of the coupled dependent variables $\theta ( x )$ , ψ(x) were used.

The third nonlinear problem with a single variable we investigate is the Poisson-Boltzmann equation. The problem herein describes the electrostatic potential u in the proximity of a uniform planar charge distribution in the presence of an electrolyte solution as function of the normal distance x from the charged surface. For a constant α determined by system properties such as the surface charge density, valency of the screening charges in the electrolyte solution, and temperature, the variational formulation corresponding to mixed boundary conditions

in a domain $x \in [ a , b ]$ is

$$
\mathcal { I } = \int _ { a } ^ { b } \left( \frac { 1 } { 2 } u ^ { \prime } ( x ) ^ { 2 } + \alpha ^ { 2 } \cosh ( u ) \right) d x - \beta u | _ { x = a } ,\tag{27}
$$

where $\beta _ { E }$ corresponds to the normalized electric field near the charged surface. The last term of Eq. 27 indicates how a Neumann condition is applied in vfNN. The corresponding ELE and the Neumann boundary condition at $x = a$ is given by

$$
{ \frac { d ^ { 2 } u } { d x ^ { 2 } } } = \alpha ^ { 2 } \sinh ( u ) , \quad { \frac { d u } { d x } } \bigg | _ { x = a } = \beta _ { E } .\tag{28}
$$

We complete the setup by choosing our parameters as $a = 0 , b = 1 , \alpha = 1 0 , \beta _ { E } = 0 . 1$ , and impose the Dirichlet boundary condition at $x = b$

$$
u | _ { x = b } = 0 .\tag{29}
$$

The results from vfNN and FEM agree very well $( \mathrm { F i g . 3 c ) }$ indicating that vFNN reliably tackles nonlinear problems with mixed boundary conditions.

Finally, we consider a nonlinear problem with two coupled dependent variables in a single functional that models a colloidal flat monolayer of monodisperse chiral hard rods. Whereas the rods are perfectly aligned in the normal direction to the membrane surface in the interior of the monolayer because of polymer-induced depletion interactions, the tilting of rods towards the membrane edge due to molecular chirality $q _ { 0 }$ (with the units of inverse length) can be treated as a packing problem with liquid crystalline symmetry, specifically as a single layer of a chiral Smectic-A $\mathrm { ( S m  – A ^ { * } ) }$ phase that melts into a cholesteric (Ch) order toward the monolayer edge [44, 45]. The rod tilting with an angle $\theta ( x )$ with respect to the monolayer surface normal manifests itself as twist deformations with a penetration depth $\lambda _ { t }$ , and the melting of ${ \mathrm { S m } } { - } \mathrm { A } ^ { * }$ into Ch can be represented by a phase field $\psi ( x )$ with a coherence length $\xi _ { | | }$ akin to the real part of the smectic order parameter. Defining a unitless parameter $\kappa \equiv \lambda _ { t } / \xi _ { | | }$ , the unitless form of the free energy nondimensionalized by $\lambda _ { t }$ in a unitless domain $x \in [ 0 , L ] \ ( x = 0$ : monolayer bulk $x = L$ : monolayer edge) is given by [44, 45]

$$
\begin{array} { c l } { \displaystyle \mathcal { I } ( \psi , \theta ) = \int _ { 0 } ^ { L } } & { \displaystyle d x \left[ - \psi ^ { 2 } + \frac { \psi ^ { 4 } } { 2 } + \frac { 1 } { \kappa ^ { 2 } } \left( \frac { d \psi } { d x } \right) ^ { 2 } \right. } \\ { \displaystyle } & { \displaystyle \left. + \psi ^ { 2 } \sin ^ { 2 } \theta + \left( \frac { d \theta } { d x } - q _ { 0 } \lambda _ { t } \right) ^ { 2 } \right] } \end{array}\tag{30}
$$

The corresponding Euler-Lagrange equations for ψ(x) and $\theta ( x )$ are

$$
{ \frac { 1 } { \kappa ^ { 2 } } } { \frac { d ^ { 2 } \psi } { d x ^ { 2 } } } + \psi \left( 1 - \sin ^ { 2 } \theta \right) - \psi ^ { 3 } = 0 ,\tag{31}
$$

$$
{ \frac { d ^ { 2 } \theta } { d x ^ { 2 } } } - { \frac { 1 } { 2 } } \psi ^ { 2 } \sin 2 \theta \ = \ 0 .\tag{32}
$$

We set the parameters as $\kappa = 1 0 , L = 1 0 , q _ { 0 } \lambda _ { t } = 0$ , and impose the following Dirichlet boundary conditions:

$$
\psi | _ { x = 0 } = 1 , \qquad \theta | _ { x = 0 } = 0 ,\tag{33}
$$

$$
\psi | _ { x = L } = 0 , \qquad \theta | _ { x = L } = \frac { \pi } { 2 } .\tag{34}
$$

To improve the convergence rate, we formulate the order parameter in terms of the NN output as $\psi ( x ) = $ $\left[ \tilde { l } ( x ) u _ { N } ( x ) + g ( x ) \right] ^ { 2 }$ , since the parity symmetry of $\psi$ evident in Eq. 30 otherwise leads to a highly degenerate loss landscape. When the rods are achiral $( q _ { 0 } \lambda _ { t } = 0 )$ twist deformations can still be observed since the rod tilting minimizes the membrane surface exposed to the depletant polymer solution [44, 45]. In our theory, the twist of achiral rods is enforced by the Dirichlet condition at $\theta | _ { x = L }$ in Eq. 34. The results are shown in Fig. 4 where the profiles from vfNN are compared to the FEM data, indicating excellent agreement for this benchmark model with more than one dependent variables, each represented by a separate neural network that has three layers and 20 neurons per layer.

## IV. DISCUSSION AND CONCLUSIONS

In this paper, we have presented the variational functional neural network (vfNN) that finds the states of a system by optimizing a physically motivated variational formulation. Many physical systems are described by an integral functional $\mathcal { I }$ as formulated in Eq. 1. The analytical extremization of $\mathcal { I }$ , i.e., $\delta { \mathcal { I } } = 0$ yields the Euler-Lagrange equations, which are the equations of motion that determine the physical states. As opposed to PINN (cf. Eq. 8), vfNN uses $\mathcal { I }$ as the error function and optimizes it with two clear strengths over PINN: first, vfNN employs characteristic physical scales that govern the relative weights of the terms in the error function $\mathcal { I }$ , instead of the ad hoc weight coeficients $\mu _ { i }$ . Second, because $\mathcal { I }$ always has lower order derivatives than the EL equations, we have found that vfNN consistently improves the computation times with respect to PINN for many functionals as fewer backpropagation steps are needed to take the lower order derivatives.

While a variational formulation provides significant computational advantage due to a loss function with lower order derivatives, stable solutions of the variations do not always exist. To address this issue, our vfNN algorithm incorporates quasi-Newton optimization based on the SR1 Hessian update paired with a stochastic line search method based on a Metropolis criterion. Our optimizer successfully converges to the first-order critical points regardless of their stability for ODEs, highlighting its potential in addressing problems where traditional optimizers, such as ADAM, may fail.

In its current state, our algorithm is agnostic to the nature of the optima, i.e., whether they are maxima, minima, saddle, or inflection points. Finding the inflection points corresponding to the “conjugate points” (Appendices A and B) is of paramount importance because of their physical significance: depending on the problem, characterizing the conjugate points can reveal dynamical or phase behavior, as well as the universal properties of a physical system. For example, in a magnetic system where magnetization is defined by a function $m ( T ) ~ ( T$ temperature), there is a ferromagnetic-to-paramagnetic transition at a critical temperature $T _ { c } .$ , which is a conjugate point. At $T = T _ { c }$ the free energy of the system, which is the error function to be optimized, has an inflection point. When $T < T _ { c } .$ , the free energy landscape develops a saddle point where it becomes nonconvex. The point $T \ = \ T _ { c }$ leads to a pitchfork bifurcation [46]. So, if the identification and characterization of conjugate points can be automated, vfNN could calculate the phase transitions of a many-body system or bifurcations and stability of a dynamical system. To find the conjugate points together with the physical solutions ˆu in a nonconvex landscape, an iterative algorithm that assesses local stability during optimization is needed.

Future work will further aim to extend vfNN to PDEs where the domain of independent variables is composed of the space and time dimensions. However, a generic vfNN PDE solver should treat space and time as independent variables on equal grounds. We will incorporate this approach to develop a general vfNN that can tackle the canonical forms of elliptic, parabolic, and hyperbolic linear PDEs before an extension to nonlinear PDEs are considered as an ultimate goal.

## ACKNOWLEDGMENTS

We thank J. Barone, S. Cheng, J. Gray, M. Pleimling for fruitful discussions and the VT College of Science for financial support. We acknowledge the VT Advanced Research Computing Center for computing resources.

## DATA AVAILABILITY

The data and codes that support the findings of this article are openly available at Ref. [42].

## Appendix A: Stability of a functional with one independent variable

To determine the stability of a functional, the second variation in $\mathcal { I }$ must be considered [34]. Here we present simplified picture using a single unknown $u ( x )$ in one dimension, $x \in [ a , b ]$ . Let variation around solution ˆu be given by $u \equiv \hat { u } + \epsilon h$ , where ϵ is a scalar and h is smooth function which obeys boundary conditions:

$$
\begin{array} { r l } { h ( x ) = 0 , \quad } & { { } x \in \partial \Omega _ { D } , } \\ { h ^ { \prime } ( x ) = 0 , \quad } & { { } x \in \partial \Omega _ { N m } . } \end{array}\tag{A1}
$$

Then, the second variation in $\mathcal { I }$ is given by,

$$
I [ h ] \equiv \frac { d ^ { 2 } J } { d \epsilon ^ { 2 } } \Bigg | _ { \epsilon = 0 } = \int _ { a } ^ { b } \Big [ \hat { F } _ { u , u } ( x ) h ^ { 2 } ( x ) + 2 \hat { F } _ { u , u ^ { \prime } } h ( x ) h ^ { \prime } ( x ) + \hat { F } _ { u ^ { \prime } , u ^ { \prime } } ( x ) h ^ { \prime 2 } ( x ) \Big ] d x .\tag{A2}
$$

For ˆu to be a minimum, $I [ h ]$ must be positive for all admissible h . This forms a secondary variational problem in h . The corresponding Euler-Lagrange equation is called Jacobi’s accessory equation [34]:

$$
\begin{array} { r c l } { { [ P ( x ) h ^ { \prime } ( x ) ] ^ { \prime } ~ = ~ Q ( x ) h ( x ) , ~ \mathrm { { \ w h e r e } } } } \\ { { } } & { { } } & { { } } \\ { { P ( x ) ~ \equiv ~ { \hat { F } } _ { u ^ { \prime } , u ^ { \prime } } ( x ) , } } \\ { { } } & { { } } & { { ~ Q ( x ) ~ \equiv ~ { \hat { F } } _ { u , u } - [ { \hat { F } } _ { u , u ^ { \prime } } ( x ) ] . } } \end{array}\tag{A3}
$$

Here, $\hat { F } _ { u , u } , \hat { F } _ { u ^ { \prime } , u ^ { \prime } }$ and $\hat { F } _ { u , u ^ { \prime } }$ denote the second functional derivatives with respect to their subscripts. Eq. A3 is a second order diferential equation in $h ( \mathbf { x } )$ with boundary conditions specified by $\operatorname { E q . \ A 1 }$ . Let $\hat { h } ( \mathbf { x } )$ be the solution to Eq. A3 satisfying boundary conditions given in A1. If 1. $P ( x ) > 0 , \forall x \in [ a , b ]$ ，

2. h<sup>ˆ</sup>(x) has no roots in $( a , b ]$ then the second variation $I [ h ]$ is positive definite; thus, uˆ is a minimum [34]. The roots of $\hat { h } ( \mathbf { x } )$ are defined as the conjugate points to the boundary point a. The three suficiency conditions that determine if ˆu is a weak minimum (defined as a minimum over the function space of continuous but not necessarily smooth functions) to the variational problem are

1. ˆu is a solution to the Euler-Lagrange equation (Eq. 3),

2. ˆu obeys the Legendre condition, $\hat { F } _ { u ^ { \prime } , u ^ { \prime } } > 0$ ∀x ∈$[ a , b ]$ ，

3. the domain of ˆu has no conjugate points to its boundaries.

## Appendix B: Locating conjugate points of functionals with a single linear ordinary Euler-Lagrange equation

We have developed Algorithm 3 to find the conjugate points for a linear ODE by using the bisection method without solving the Jacobi’s accessory equation, which requires knowing ˆu in the first place. An example for the extremization of a functional from this algorithm is shown in Fig. 2c.

This algorithm has two significant limitations: first, we assume that Legendre condition is satisfied while looking for the conjugate points. Second, it only applies to the linear ODEs since their solution is unique for the given boundary condition. Hence, the solution to the variational problem in the given domain is either a minimum or not a minimum, and this property is exploited to find the conjugate points.

```csv
Algorithm 3: Finding conjugate points
Input: {x<sub>start</sub>, x<sub>end</sub>}: domain, J : loss functional, δ<sub>1</sub>:
tolerance, δ : tolerance for convergence of
vfNN, N: epochs
1 x<sub>left</sub> ← x<sub>start</sub>, x<sub>c,list</sub> ← [ ];
2 Optimize vfNN on x ∈ [x<sub>left</sub>, x<sub>end</sub>] using ADAM for N
epochs;
3 if $\Delta \mathcal { I } / \mathcal { I } < \delta _ { 2 }$ then
4 return x<sub>c,list</sub>;
5 else
6 repeat
7 x<sub>a</sub> ← x<sub>left</sub>, x<sub>b</sub> ← x<sub>end</sub>;
8 repeat
9 $x _ { c } \gets ( x _ { a } + x _ { b } ) / 2 ;$
10 Optimize vfNN on x ∈ [x , x<sub>c</sub>] using
ADAM for N epochs;
11 if $\Delta \mathcal { I } / \mathcal { I } < \delta _ { 2 }$ then
12 x<sub>a</sub> ← x<sub>c</sub>;
13 else
14 x<sub>b</sub> ← x<sub>c</sub>;
15 until $x _ { b } - x _ { a } < \delta _ { 1 } ;$
16 Append (x<sub>a</sub> + x<sub>b</sub>)/2 to x<sub>c,list</sub>;
17 x<sub>left</sub> ← x<sub>c</sub>;
18 until $\Delta \mathcal { I } / \mathcal { I } < \delta _ { 2 } ;$
19 return x<sub>c,list</sub>;
```

[1] K. Hornik, M. Stinchcombe, and H. White, Multilayer feedforward networks are universal approximators, Neural networks 2, 359 (1989).

[2] L. B. Rall and G. F. Corliss, An introduction to automatic diferentiation, Computational Diferentiation: Techniques, Applications, and Tools 89, 1 (1996).

[3] M. Raissi, P. Perdikaris, and G. Karniadakis, Physicsinformed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations, Journal of Computational Physics 378, 686 (2019).

[4] J. D. Toscano, V. Oommen, A. J. Varghese, Z. Zou, N. Ahmadi Daryakenari, C. Wu, and G. E. Karniadakis, From pinns to pikans: recent advances in physicsinformed machine learning, Machine Learning for Computational Science and Engineering 1, 15 (2025).

[5] B. Zhang, G. Wu, Y. Gu, X. Wang, and F. Wang, Multi-domain physics-informed neural network for solving forward and inverse problems of steady-state heat conduction in multilayer media, Physics of Fluids 34, 116116 (2022), https://pubs.aip.org/aip/pof/articlepdf/doi/10.1063/5.0116038/16602534/116116 1 online.pd

[6] T. Hu, B. Jin, and Z. Zhou, Solving poisson problems in polygonal domains with singularity enriched physics informed neural networks, SIAM Journal on Scientific Computing 46, C369 (2024), https://doi.org/10.1137/23M1601195.

[7] N. Ahmadi Daryakenari, M. De Florio, K. Shukla, and G. E. Karniadakis, Ai-aristotle: A physics-informed framework for systems biology gray-box identification, PLOS Computational Biology 20, 1 (2024).

[8] A. Yazdani, L. Lu, M. Raissi, and G. E. Karniadakis, Systems biology informed deep learning for inferring parameters and hidden dynamics, PLOS Computational Biology 16, 1 (2020).

[9] H. Jo, H. Hong, H. J. Hwang, W. Chang, and J. K. Kim, Density physics-informed neural networks reveal sources of cell heterogeneity in signal transduction, Patterns 5, 100899 (2024).

[10] Y. Qian, G. Zhu, Z. Zhang, S. Modepalli, Y. Zheng, X. Zheng, G. Frydman, and H. Li, Coagulo-Net: Enhancing the mathematical modeling of blood coagulation using physics-informed neural networks, Neural Netw 180, 106732 (2024).

[11] X. Jin, S. Cai, H. Li, and G. E. Karniadakis, Nsfnets (navier-stokes flow nets): Physics-informed neural networks for the incompressible navier-stokes equations, Journal of Computational Physics 426, 109951 (2021).

[12] H. Eivazi, M. Tahani, P. Schlatter, and R. Vinuesa, Physics-informed neural networks for solving reynoldsaveraged navier–stokes equations, Physics of Fluids 34, 075117 (2022), https://pubs.aip.org/aip/pof/articlepdf/doi/10.1063/5.0095270/19816179/075117 1 online.pd

[13] A. D. Jagtap, Z. Mao, N. Adams, and G. E. Karniadakis, Physics-informed neural networks for inverse problems in supersonic flows, Journal of Computational Physics 466, 111402 (2022).

[14] M. De Florio, E. Schiassi, B. D. Ganapol, and R. Furfaro, Physics-informed neural networks for rarefied-gas dynamics: Thermal creep flow in the bhatnagar–gross–krook approximation, Physics of Fluids

33, 047110 (2021), https://pubs.aip.org/aip/pof/article pdf/doi/10.1063/5.0046181/13636200/047110 1 online.pdf.

[15] W. Li, M. Z. Bazant, and J. Zhu, A physics-guided neural network framework for elastic plates: Comparison of governing equations-based and energy-based approaches, Computer Methods in Applied Mechanics and Engineering 383, 113933 (2021).

[16] J.-H. Bastek and D. M. Kochmann, Physics-informed neural networks for shell structures, European Journal of Mechanics - A/Solids 97, 104849 (2023).

[17] U. b. Waheed, E. Haghighat, T. Alkhalifah, C. Song, and Q. Hao, Pinneik: Eikonal solution using physics-informed neural networks, Computers and amp; Geosciences 155, 104833 (2021).

[18] L. F. Nazari, E. Camponogara, and L. O. Seman, Physics-informed neural networks for modeling water flows in a river channel, IEEE Transactions on Artificial Intelligence 5, 1001 (2024).

[19] Z. Wu, H. Wang, C. He, B. Zhang, T. Xu, and Q. Chen, The application of physics-informed machine learning in multiphysics modeling in chemical engineering, Industrial & Engineering Chemistry Research 62, 18178 (2023), https://doi.org/10.1021/acs.iecr.3c02383.

[20] E. Kharazmi, Z. Zhang, and G. E. Karniadakis, Variational physics-informed neural networks for solving partial diferential equations (2019), arXiv:1912.00873 [cs.NE].

[21] E. Kharazmi, Z. Zhang, and G. E. Karniadakis, hpvpinns: Variational physics-informed neural networks with domain decomposition, Computer Methods in Applied Mechanics and Engineering 374, 113547 (2021).

[22] S. Berrone, C. Canuto, and M. Pintore, Variational physics informed neural networks: the role of quadratures and test functions, Journal of Scientific Computing 92, 100 (2022).

[23] Y. Zang, G. Bao, X. Ye, and H. Zhou, Weak adversarial networks for high-dimensional partial diferential equations, Journal of Computational Physics 411, 109409 (2020).

[24] R. Khodayi-Mehr and M. Zavlanos, Varnet: Variational neural networks for the solution of partial diferential equations, in Proceedings of the 2nd Conference on Learning for Dynamics and Control, Proceedings of Machine Learning Research, Vol. 120, edited by A. M. Bayen, A. Jadbabaie, G. Pappas, P. A. Parrilo, B. Recht, C. Tomlin, and M. Zeilinger (PMLR, 2020) pp. 298–307.

[25] W. E and B. Yu, The deep ritz method: A deep learningbased numerical algorithm for solving variational problems, Communications in Mathematics and Statistics 6, 1 (2018).

[26] M. Liu, Z. Cai, and K. Ramani, Deep ritz method with adaptive quadrature for linear elasticity, Computer Methods in Applied Mechanics and Engineering 415, 116229 (2023).

[27] C. Uriarte, D. Pardo, I. Muga, and J. Mu˜noz-Matute, A deep double ritz method (d2rm) for solving partial diferential equations using neural networks, Computer Methods in Applied Mechanics and Engineering 405, 115892 (2023).

[28] J. Lu, Y. Lu, and M. Wang, A priori generalization analysis of the deep ritz method for solving high dimensional

elliptic equations (2021), arXiv:2101.01708 [math.NA].

[29] E. Samaniego, C. Anitescu, S. Goswami, V. Nguyen-Thanh, H. Guo, K. Hamdia, X. Zhuang, and T. Rabczuk, An energy approach to the solution of partial diferential equations in computational mechanics via machine learning: Concepts, implementation and applications, Computer Methods in Applied Mechanics and Engineering 362, 112790 (2020).

[30] V. M. Nguyen-Thanh, C. Anitescu, N. Alajlan, T. Rabczuk, and X. Zhuang, Parametric deep energy approach for elasticity accounting for strain gradient efects, Computer Methods in Applied Mechanics and Engineering 386, 114096 (2021).

[31] S. Karumuri, R. Tripathy, I. Bilionis, and J. Panchal, Simulator-free solution of high-dimensional stochastic elliptic partial diferential equations using deep neural networks, Journal of Computational Physics 404, 109120 (2020).

[32] V. M. Nguyen-Thanh, X. Zhuang, and T. Rabczuk, A deep energy method for finite deformation hyperelasticity, European Journal of Mechanics-A/Solids 80, 103874 (2020).

[33] H. Sheng and C. Yang, Pfnn: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries, Journal of Computational Physics 428, 110085 (2021).

[34] I. M. Gelfand, R. A. Silverman, et al., Calculus of variations (Courier Corporation, 2000).

[35] L. Lu, R. Pestourie, W. Yao, Z. Wang, F. Verdugo, and S. G. Johnson, Physics-informed neural networks with hard constraints for inverse design, SIAM Journal on Scientific Computing 43, B1105 (2021), https://doi.org/10.1137/21M1397908.

[36] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization (2017), arXiv:1412.6980 [cs.LG].

[37] D. C. Liu and J. Nocedal, On the limited memory bfgs method for large scale optimization, Mathematical programming 45, 503 (1989).

[38] J. Nocedal and S. J. Wright, Numerical optimization (Springer, 1999).

[39] S. Chib and E. Greenberg, Understanding the metropolishastings algorithm, The american statistician 49, 327 (1995).

[40] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, Optimization by simulated annealing, Science 220, 671 (1983).

[41] D. Bertsimas and J. Tsitsiklis, Simulated annealing, Statistical science 8, 10 (1993).

[42] For data and codes, see https://github.com/nadirkaplan/ variational-functional-neural-networks/.

[43] M. S. Alnæs, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg, C. Richardson, J. Ring, M. E. Rognes, and G. N. Wells, The FEniCS project version 1.5, Archive of Numerical Software 3, 10.11588/ans.2015.100.20553 (2015), software version 2019.1.0.

[44] C. N. Kaplan and R. B. Meyer, Colloidal membranes of hard rods: unified theory of free edge structure and twist walls, Soft Matter 10, 4700 (2014).

[45] M. J. Zakhary, T. Gibaud, C. N. Kaplan, E. Barry, R. Oldenbourg, R. B. Meyer, R. D. Kamien, R. A. Pelcovits, T. R. Powers, and Z. Dogic, Imprintable membranes from incomplete chiral coalescence, Nature Communications 5, 3063 (2014).

[46] S. H. Strogatz, Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering, 2nd ed. (CRC Press, 2015).