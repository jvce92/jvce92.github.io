---
title: 'Simulating Poisson Networks Efficiently'
date: 2018-08-31
permalink: /posts/2012/08/simulating-poisson-networks-efficiently/
tags:
  - Poisson Point Process
  - Wireless Networks
  - Simulation
---

# Simulating Poisson Networks Efficiently


When I first started my Ph.D. I learned about modeling large wireless network using point processes. They allow you to obtain tractable mathematical models that scale well with the size of the network. The most commonly used model in wireless networks is the homogeneous Poisson point proces (PPP), because it has a reasonable assumption about the network geometry and leads to tractable models. To briefly introduce the concept, say we have a set of two dimensional points {% raw %} $\phi = \{u_1, \cdots\}$ {% endraw %} randomly distributed in $\mathbb{R}^2$, such that for every compact set $B \in \mathbb{R}^2$. The set of points $\phi$ is a HPPP if: 

* The number of points of $\phi$ in $B$ (denoted by $N(B)$) are poisson distributed with rate $\lambda |B|$, where $\lambda$ is the intensity of the HPPP and $|B|$ is the are of $B$.
* If $B_1$,$B_2$, ..., $B_m$ are all disjoint sets in $\mathbb{R}^2$, then $N(B_1)$,$N(B_2)$, ..., $N(B_m)$ are independent random variables

For more details on the theory behind this I recommending reading the book [*Stochastic Geometry for Wireless Networks*](https://www.amazon.ca/Stochastic-Geometry-Wireless-Networks-Professor/dp/1107014697/ref=sr_1_1?ie=UTF8&qid=1535753229&sr=8-1&keywords=stochastic+geometry+for+wireless+networks), by Martin Haenggi.

In order to simulate a HPPP, we determine an area of interest, say the ball $b(0,R)$, and generate $N$ points inside, where $N$ is drawn from a poisson distribution with rate $\lambda \pi R^2$. An example of a function to generate a realization of a HPPP can be seem below:


```python
import numpy as np

def generate_PPP(lam, radius):
    # Number of points inside a ball
    N = np.random.poisson(lam * np.pi * radius**2)
    # Distance from the origin
    r = radius * np.sqrt(np.random.random(N))
    # Angle from the origin
    ang = 2 * np.pi * np.random.random(N)
    # Matrix of the PPP
    phi = np.vstack((r * np.cos(ang), r * np.sin(ang))).T
    return phi
```

Usually we run a simulation of a HPPP to validate the analytical results. When simulating Poisson networks, we are usually interested in averaging a performance metric, which depends on the distance distribution between the points of a HPPP and the origin or between two different HPPP's, through a Monte Carlo simulation. This means we have to calculate distances quite a few times. 

## Naive Distance Calculation

When I first started simulating large networks, my simulations took a long time to run. Let's say you have to calculate the distances between two point processes $\phi_1$ and $\phi_2$ (e.g. a user point process and a base station point process), each with $N_1$ and $N_2$ points respectively. If you do it naively like I used to, you will have a nested for loop, looping through every point of each process, resulting in a complexity of $\mathcal{O}(N_1 N_2)$, as the function shown below


```python
def naive_distance(phi_1, phi_2):
    N1, _ = phi_1.shape
    N2, _ = phi_2.shape
    D = np.zeros((N1, N2))
    for n1 in range(N1):
        for n2 in range(N2):
            D[n1, n2] = np.sqrt((phi_1[n1, 0] - phi_2[n2, 0])**2 + (phi_1[n1, 1] - phi_2[n2, 1])**2)
            
    return D
```

## More Efficient Approach

Once I started simulating HPPP's with higher densities this simulations started taking hours. I did some research on how to speed them up and figured out we can calculate these distances more efficiently. The distance between points $\mathbf{x} \in \phi_1$ and $\mathbf{y} \in \phi_2$ is given by

\begin{equation}
D_{\mathbf{x}, \mathbf{y}} = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2} = \sqrt{x_1^2 + x_2^2 + y_1^2 + y_2^2 - 2 (x_1 y_1 + x_2 y_2)}
\end{equation}

We can obtain a matrix $\mathbf{D} \in \mathbb{R}^{N_1 \times N_2}$ where each coordinate $D_{i, j}$ is the distance between the $i$-th point in $\phi_1$ to the $j$-th point in $\phi_2$ by

\begin{equation}
D = (\mathbf{\Phi_1} \mathbf{1}_{2 \times 1})^T \mathbf{1}_{1 \times N_2} + (\mathbf{\Phi_2} \mathbf{1}_{2 \times 1})^T \mathbf{1}_{1 \times N_1} - 2 \mathbf{\Phi_1} \mathbf{\Phi_2}^T
\end{equation}

Where $\mathbf{\Phi_1}$ and $\mathbf{\Phi_2}$ are a $N_1 \times 2$ and a $N_2 \times 2$ matrix respectively, where each row is a point from $\phi_1$ and $\phi_2$. The vector/matrix $\mathbf{1}_{i \times j}$ is a $i \times j$ vector/matrix. This expression involves two matrix-vector multiplications, two inner products and one matrix-matrix multiplication. It might seem that the resulting complexity of this second algorithm is larger than the naive one, but processors and linear algebra libraries are optimized by experienced engineers to execute linear algebra operations really fast, which results in a smaller running time than the naive algorithm. 

A sample function that calculates the distance by this method is shown below


```python
def efficient_distance(phi_1, phi_2):
    D = (np.sum(phi_1**2, axis=1).reshape((-1, 1))).dot(np.ones((1, phi_2.shape[0]))) + \
        (np.sum(phi_2**2, axis=1).reshape((-1, 1))).dot(np.ones((1, phi_1.shape[0]))).T - \
        2 * phi_1.dot(phi_2.T)
    D = np.sqrt(D)
    
    return D
```

## Comparison

Now let's compare the running time of both functions


```python
lam = 400
radius = 1
phi_1 = generate_PPP(lam, radius)
phi_2 = generate_PPP(lam, radius)
print("Size of phi_1: {0}".format(phi_1.shape))
print("Size of phi_2: {0}".format(phi_2.shape))
```

    Size of phi_1: (1250, 2)
    Size of phi_2: (1316, 2)



```python
%timeit d_naive = naive_distance(phi_1, phi_2)
```

    5.14 s ± 493 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%timeit d_eff = efficient_distance(phi_1, phi_2)
```

    46.4 ms ± 1.43 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


As you can see, on average, the the efficient approach was 110.78 times faster than the naive one.

You can download a Jupyter notebook with this tutorial [here](https://jvce92.github.io/files/simulating_poisson_networks.ipynb)
