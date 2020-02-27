# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:39:35 2019

@author: MONSTER VISION
"""

import tensorflow as tf
meta_path='D:/checkpoint/model.ckpt-30776.meta'
#meta_path = 'model.ckpt-22480.meta'      # Your .meta file
output_node_names = ['softmax_tensor']    #Name of Output nodes

with tf.Session() as sess:
    # Restore the graph
    
    """
    --> Recreates the Saved grpah from meta file & returns a saver constructed.
    --> Exporting/importing meta graphs is not supported.
    --> No graph exists when eager execution is enabled.
    
    """
    
    
    saver = tf.train.import_meta_graph(meta_path)


    # Load weights
    """
    --> The tf.train.Saver object not only saves variables to checkpoint files,it also restores variables. 
    --> Note that when you restore variables you do not have to initialize them beforehand.
    --> 'tf.train.Saver.restore' method to restore variables from the checkpoint files
    """
    
    """ 
    --> Restores previously saved variables.
    --> This method runs the ops added by the constructor for restoring variables.
    --> It requires a session in which the graph was launched. 
    --> The variables to restore do not have to have been initialized, 
    as restoring is itself a way to initialize variables.
    """


    saver.restore(
            sess,#A Session to use to restore the parameters. None in eager mode.
            tf.train.latest_checkpoint('D:/checkpoint/')#Finds the filename of latest saved checkpoint file.
            )



    # Freeze the graph
    """
    --> If You Don't know the Output node name then use fallowing line.
    It will take all the nodes as output ones as shown below.
    --> But  it's unusual situation,
    because if you don't know the output node, you cannot use the graph actually.
    """
#    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    """ 
    --> Replaces all the variables in a graph with constants of the same values.
    --> Returns GraphDef containing a simplified version of the original.
    """

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,               #Active Session
            sess.graph_def,     #GraphDef object holding the network.
            output_node_names   #List of name strings for the result nodes of the graph.
            )

    
    
    # Save the frozen graph
    with open('D:/checkpoint/output_graph.pb', 'wb') as f:
        #wb=write binary mode
        f.write(frozen_graph_def.SerializeToString())
      
        #SerializeToString(): serializes the message and returns it as a string. 
        #Note that the bytes are binary, not text;
        #we only use the str type as a convenient container.
      
      



