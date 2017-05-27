# Getting started with tensorflow
# Done by James Lemayian

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also float32 implicitly

# This does not print the actual value of the nodes because the nodes are not evaluated
print(node1,node2)

# Running the nodes in a session actually evaluates them
sess = tf.Session()
print(sess.run([node1,node2]))

# Adding two nodes using the tf add operation
node3 = tf.add(node1, node2)

print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

# Parametizing graphs to accept external inputs
# a placeholder is a promise to provide a value later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adderNode = a + b # + sign provides a shortcut for tf.add(a,b)

print(sess.run(adderNode, {a:3, b:4.5}))
print(sess.run(adderNode, {a:[1,3], b: [4,4.5]}))

# complicate stuff, maybe add and tripple
addAndTripple = adderNode * 3.

print(sess.run(addAndTripple, {a: 3., b: 4.5}))

# How to take arbitrary value using a Variables
# Variables allow us to add trainable  parameters to a graph
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linearModel = W * x + b

# Initializing the global variables
sess.run(tf.global_variables_initializer())

# evaluate linear model for several values of x as follows
print(sess.run(linearModel, {x:[1,2,3,4]}))

# evaluating model on training data
y = tf.placeholder(tf.float32)
squaredDeltas = tf.square(linearModel - y)
loss = tf.reduce_sum(squaredDeltas)

print(sess.run(loss, {x:[1,2,3,4], y: [0,-1,-2,-3]}))

# Improving it manually by fixing the values of x and b
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y: [0,-1,-2,-3]}))
