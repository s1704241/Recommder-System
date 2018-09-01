# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:10:24 2018

@author: hasee
"""
#!/usr/bin/env python

"""A simple example of the sum-product algorithm with 2 factors: $fa$ being 
the prior to the Gaussian $x1$ and $fb$ being the evidence (from the user 
response) for the same variable. 

This is a simple example of the sum-product algorithm (belief propagation)
on a factor graph with a Gaussian random variable.

      /--\      +----+      /--\
     | fa |-----| x1 |-----| fb |
      \--/      +----+      \--/ 

"""

from fglib import graphs, nodes, inference, rv


class Belief_Updating:
    def __init__(self, responseVariance):
        #Assumed response variance on user response
        self.responseVariance = responseVariance

    def __call__(self, priorMean, priorVariance, userResponse):
    # Create factor graph
        fg = graphs.FactorGraph()
        
        # Create variable nodes
        x1 = nodes.VNode("x1", rv.Gaussian)
        x2 = nodes.VNode("x2", rv.Gaussian)
        
        # Create factor nodes (with joint distributions)
        priorMean = 3;
        priorVariance = 0.1;
        priorPrecision = 1/priorVariance;
        priorPrecisionMean = priorPrecision * priorMean;
        responsePrecision = 1/self.responseVariance;
        responsePrecisionMean = responsePrecision * userResponse;
        
        fa = nodes.FNode( "fa", rv.Gaussian.inf_form( [priorPrecision], [priorPrecisionMean], x1) )
        fb = nodes.FNode( "fb", rv.Gaussian.inf_form( [responsePrecision], [responsePrecisionMean], x1) )
        
        # Add nodes to factor graph
        fg.set_nodes([x1])
        fg.set_nodes([fa, fb])
        
        # Add edges to factor graph
        fg.set_edge(fa, x1)
        fg.set_edge(x1, fb)
        
        # Perform sum-product algorithm on factor graph
        # and request belief of variable node x4
        belief = inference.sum_product(fg, x1)
        
        # Print belief of variables
        #Belief of variable node x1
        variance = 1 / belief._W
        mean = belief._Wm / belief._W
        
        return [mean,variance]
        
        """The below will not work, because the procedure for 
        inverting the precision matrix checks that it has at 
        least 2 dimensions. Using the default constructor for 
        the Gaussian variable will not work for the same 
        reason.
        """
        #print("Belief of variable node x1:")
        #print(x1.belief())
        
        #print("Belief of variable node x1:")
        #print(x1.belief(normalize=True))
        
        #print("Unnormalized belief of variable node x1:")
        #print(x1.belief(normalize=False))