# Demonstrating the Math Behind the PageRank Algorithm
This project used the Python [NumPy](https://www.numpy.org/) and [NetworkX](https://networkx.github.io/) libraries to demonstrathe the mathematical operations behind the PageRank network analysis algorithm which was used to create Google's search engine. The algorithm can be intuitvely understood as using the probabilities of using any link on a webpage to go to other webpages as a means of evaluating a webpage's overall importance in the world wide web. In this project, a very simple toy network is used to test the custom functions and to compare their results in compariong to the equivalent NetworkX functions.

## Toy Network
Below is the list of edges (in the form of tuples) which represent a directional connection from the node specified on the left side of the tuple toward the node specified on the right side of the tuple. A visualization of this toy network can be found in the [Jupyter Notebook](https://github.com/Yoseph-Zuskin/PageRank_Algorithm/blob/master/Social_Network_Analysis_Assignment.ipynb).

```
[('9','7'),('8','7'),('8','6'),('8','5'),('7','6'),('7','5'),('6','5'),('6','4'),('5','4'),('4','3'),('4','1'),('8','7'),('3','2'),('3','1'),('2','1')]
```

## Transition Matrix
The key mathematical tool used in this algorithm to rank webpages is a matrix comprised of values that correspond to the probability of visiting a certain webpage given that you are on a webpage with a limited number of external links, or none at all. There are two important additional consideration that need to be taken into account when creating the transition matrix. The first is known as the dangeling node problem, which occurs when a webpage has no external links (such as '1' in this toy network), and is addressed by replacing the zero probability of visiting all webpages from this dead-end webpage with an equal probability of visiting any other webpage (1/9 in this case). The second problem is known as the random surfer problem, which exists due to the possibility of a webpage visitor ignoring the links to other webpages and randomly going to another webpage. In this transition matrix, every column corresponds to the webpage nodes in the toy network in sorted sequence ('1' to '9'), which each element in each column representing the probability of visiting a webpage linked to the webpage node.

The initial tranition matrix, without considering the two problems, would look like this:
```
matrix([[0.        , 1.        , 0.5       , 0.5       , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.5       , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.5       , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 1.        ,
         0.5       , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.5       , 0.5       , 0.33333333, 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.5       , 0.33333333, 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.33333333, 1.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.        ]])
```

The initial tranition matrix, without considering these problems, would look like this:
```
matrix([[0.11111111, 1.        , 0.5       , 0.5       , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.11111111, 0.        , 0.5       , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.11111111, 0.        , 0.        , 0.5       , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.11111111, 0.        , 0.        , 0.        , 1.        ,
         0.5       , 0.        , 0.        , 0.        ],
        [0.11111111, 0.        , 0.        , 0.        , 0.        ,
         0.5       , 0.5       , 0.33333333, 0.        ],
        [0.11111111, 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.5       , 0.33333333, 0.        ],
        [0.11111111, 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.33333333, 1.        ],
        [0.11111111, 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.        ],
        [0.11111111, 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.        ]])
```

If we assume a 15% probability of randomly visiting any webpage rather than going through a link on a webpage, then a dampening factor (d) of 0.85 needs to be applied to the above transition matrix (M) through the following operation (where I is the initial vector which is discussed below):
```
d*M+(1-d)*I
```

The above operation would produce the following transition matrix:
```
matrix([[0.11111111, 0.86666667, 0.44166667, 0.44166667, 0.01666667,
         0.01666667, 0.01666667, 0.01666667, 0.01666667],
        [0.11111111, 0.01666667, 0.44166667, 0.01666667, 0.01666667,
         0.01666667, 0.01666667, 0.01666667, 0.01666667],
        [0.11111111, 0.01666667, 0.01666667, 0.44166667, 0.01666667,
         0.01666667, 0.01666667, 0.01666667, 0.01666667],
        [0.11111111, 0.01666667, 0.01666667, 0.01666667, 0.86666667,
         0.44166667, 0.01666667, 0.01666667, 0.01666667],
        [0.11111111, 0.01666667, 0.01666667, 0.01666667, 0.01666667,
         0.44166667, 0.44166667, 0.3       , 0.01666667],
        [0.11111111, 0.01666667, 0.01666667, 0.01666667, 0.01666667,
         0.01666667, 0.44166667, 0.3       , 0.01666667],
        [0.11111111, 0.01666667, 0.01666667, 0.01666667, 0.01666667,
         0.01666667, 0.01666667, 0.3       , 0.86666667],
        [0.11111111, 0.01666667, 0.01666667, 0.01666667, 0.01666667,
         0.01666667, 0.01666667, 0.01666667, 0.01666667],
        [0.11111111, 0.01666667, 0.01666667, 0.01666667, 0.01666667,
         0.01666667, 0.01666667, 0.01666667, 0.01666667]])
```

## Initial Vector
The transition matrix shown above will be iteratively multiplied with itself and finally the initial vector to generate the webpage rank score. This initial matrix represents the probability of visiting any webpage in the toy network, which will be adjusted through this iterative mathematical process. In this toy network, it will be as follow:
```
matrix([[0.11111111],
        [0.11111111],
        [0.11111111],
        [0.11111111],
        [0.11111111],
        [0.11111111],
        [0.11111111],
        [0.11111111],
        [0.11111111]])
```

## The Algorithm
Using the tranisiton matrix (M) and initial vector (I), we can adjust the rank scores by multiplying them together as:
```
M**n*I
```
Where n is the number of times the transition matrix needs to be multipled by itself in order to achieve convergence. In the custom function created in the [Jupyter Notebook](https://github.com/Yoseph-Zuskin/PageRank_Algorithm/blob/master/Social_Network_Analysis_Assignment.ipynb), the verbose parameter provided the option to show how many iterations is required until the resulting pageranks and not any different then they are will one less iteration. In this toy network example, the convergence will occur at the 60th iteration when using a dampening factor of 0.85, with the following ranks for each webpage rank:
```
{'1': 0.2160493827160494,
 '2': 0.07438271604938271,
 '3': 0.07438271604938271,
 '4': 0.16882716049382718,
 '5': 0.1530864197530864,
 '6': 0.10586419753086419,
 '7': 0.15308641975308643,
 '8': 0.027160493827160494,
 '9': 0.027160493827160494}
 ```
