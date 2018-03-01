import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import random

LIVERPOOL_ID = 8650
SCHALKE_ID = 10189
PSG_ID = 9847

LEARNING_RATE = .8
TEST_SIZE = 0.2
BATCH_SIZE = 120
NUM_EPOCHS = 800
NUM_TASKS = 3

NUM_FEATURES = 8
HIDDEN_NEURONS = 8
NUM_OUTPUTS = 3 # Win, Tie, and Loss

conn = sqlite3.connect("../soccer.sqlite")

def get_team_matches(team_id, conn):
    query = """
    SELECT season, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal
    FROM Match
    WHERE home_team_api_id = %s OR away_team_api_id = %s
    """ % (team_id, team_id)

    # Order so that more recent matches come first.
    return pd.read_sql_query(query, conn)[::-1]

def get_team_stats(team_id, conn):
    query = """
    SELECT date, buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing,
    chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth
    FROM Team_Attributes WHERE team_api_id = %s
    ORDER BY date DESC
    """ % (team_id)

    return pd.read_sql_query(query, conn)

def get_team_training_data(team_id, conn, cleaned=False):
    # Get all home matches for the given team.
    home_query = """
    SELECT
        M.home_team_goal AS own_goal, M.away_team_goal AS opponent_goal,
        T.buildUpPlaySpeed, T.buildUpPlayPassing, T.chanceCreationPassing, T.chanceCreationCrossing,
        T.chanceCreationShooting, T.defencePressure, T.defenceAggression, T.defenceTeamWidth, M.date, T.team_long_name AS opponent_name
    FROM
        Match AS M
    LEFT JOIN
        Avg_Team_Attributes AS T ON T.team_api_id = M.away_team_api_id
    WHERE
        M.home_team_api_id = %s
    """ % (team_id)
    
    
    away_query = """
    SELECT
        M.home_team_goal AS opponent_goal, M.away_team_goal AS own_goal,
        T.buildUpPlaySpeed, T.buildUpPlayPassing, T.chanceCreationPassing, T.chanceCreationCrossing,
        T.chanceCreationShooting, T.defencePressure, T.defenceAggression, T.defenceTeamWidth, M.date, T.team_long_name AS opponent_name
    FROM
        Match AS M
    LEFT JOIN
        Avg_Team_Attributes AS T on T.team_api_id = M.home_team_api_id
    WHERE
        M.away_team_api_id = %s
    """ % (team_id)

    home_matches = pd.read_sql_query(home_query, conn)
    away_matches = pd.read_sql_query(away_query, conn)
    
    frames = [home_matches, away_matches]

    combined = pd.concat(frames)
    
    labels = pd.DataFrame()
    labels['win'] = np.where(combined['own_goal'] > combined['opponent_goal'], 1, 0)
    labels['tie'] = np.where(combined['own_goal'] == combined['opponent_goal'], 1, 0)
    labels['loss'] = np.where(combined['own_goal'] < combined['opponent_goal'], 1, 0)

    if cleaned:
        combined.drop(['opponent_goal', 'own_goal', 'date', 'opponent_name'], axis=1, inplace=True)
    
    return (combined.as_matrix(), labels.as_matrix())

def get_mini_batch(data, labels, size):
    indices = random.sample(range(data.shape[0]), size)
    mini_data = data[indices]
    mini_labels = labels[indices]

    return (mini_data, mini_labels)


liverpool, liverpool_labels = get_team_training_data(LIVERPOOL_ID, conn, cleaned=True)
schalke, schalke_labels = get_team_training_data(SCHALKE_ID, conn, cleaned=True)
psg, psg_labels = get_team_training_data(PSG_ID, conn, cleaned=True)

# Scale the data to have mean 0, variance 1 (unit variance).

scaler = StandardScaler()
# Fit the scaler on all three data sets.
scaler.partial_fit(liverpool)
scaler.partial_fit(schalke)
scaler.partial_fit(psg)

liverpool = scaler.transform(liverpool)
schalke = scaler.transform(schalke)
psg = scaler.transform(psg)

# Split the data into training sets and test sets.

liverpool_train, liverpool_test, liverpool_train_labels, liverpool_test_labels = train_test_split(liverpool, liverpool_labels, test_size=TEST_SIZE)
schalke_train, schalke_test, schalke_train_labels, schalke_test_labels = train_test_split(schalke, schalke_labels, test_size=TEST_SIZE)
psg_train, psg_test, psg_train_labels, psg_test_labels = train_test_split(psg, psg_labels, test_size=TEST_SIZE)

# Placeholders for inputs to the neural network.

X = tf.placeholder(tf.float32, [None, NUM_FEATURES], name="X")

T1 = tf.placeholder(tf.float32, [None, NUM_OUTPUTS], name="T1_labels")
T2 = tf.placeholder(tf.float32, [None, NUM_OUTPUTS], name="T2_labels")
T3 = tf.placeholder(tf.float32, [None, NUM_OUTPUTS], name="T3_labels")

# Weights for the various layers of the neural network.

initial_shared_weights = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS)
initial_T1_weights = np.random.rand(HIDDEN_NEURONS, NUM_OUTPUTS)
initial_T2_weights = np.random.rand(HIDDEN_NEURONS, NUM_OUTPUTS)
initial_T3_weights = np.random.rand(HIDDEN_NEURONS, NUM_OUTPUTS)

shared_weights = tf.Variable(initial_shared_weights, dtype=tf.float32, name="shared_weights")
T1_weights = tf.Variable(initial_T1_weights, dtype=tf.float32, name="T1_weights")
T2_weights = tf.Variable(initial_T2_weights, dtype=tf.float32, name="T2_weights")
T3_weights = tf.Variable(initial_T3_weights, dtype=tf.float32, name="T3_weights")

# Construct the shared and task-specific layers.
shared_layer = tf.sigmoid(tf.matmul(X, shared_weights))

T1_layer = tf.matmul(shared_layer, T1_weights)
T2_layer = tf.matmul(shared_layer, T2_weights)
T3_layer = tf.matmul(shared_layer, T3_weights)

# Define loss functions and optimizers for each of the tasks.
# Softmax activation on outputs because we want to perform multi-class, mutually exclusive classification.
T1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T1, logits=T1_layer))
T2_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T2, logits=T2_layer))
T3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T3, logits=T3_layer))

T1_op = tf.train.AdamOptimizer().minimize(T1_loss)
T2_op = tf.train.AdamOptimizer().minimize(T2_loss)
T3_op = tf.train.AdamOptimizer().minimize(T3_loss)

# Train the neural network.
T1_errors = []
T2_errors = []
T3_errors = []

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for epoch in range(NUM_EPOCHS):
    # Choose mini batches of training data for Stochastic Gradient Descent.
    
    liverpool_batch, liverpool_batch_labels = get_mini_batch(liverpool_train, liverpool_train_labels, BATCH_SIZE)
    
    _, t1, T1_output = sess.run([T1_op, T1_loss, tf.nn.softmax(T1_layer)], {X: liverpool_batch, T1: liverpool_batch_labels})
    
    schalke_batch, schalke_batch_labels = get_mini_batch(schalke_train, schalke_train_labels, BATCH_SIZE)
    
    _, t2, T2_output = sess.run([T2_op, T2_loss, tf.nn.softmax(T2_layer)], {X: schalke_batch, T2: schalke_batch_labels})
    
    psg_batch, psg_batch_labels = get_mini_batch(psg_train, psg_train_labels, BATCH_SIZE)
    
    _, t3, T3_output = sess.run([T3_op, T3_loss, tf.nn.softmax(T3_layer)], {X: psg_batch, T3: psg_batch_labels})

    T1_errors.append(t1)
    T2_errors.append(t2)
    T3_errors.append(t3)

# Plot the error over epochs.
plt.plot(range(0, len(T1_errors)), T1_errors, "r", label="Liverpool")
plt.plot(range(0, len(T2_errors)), T2_errors, "g", label="Schalke")
plt.plot(range(0, len(T3_errors)), T3_errors, "b", label="PSG")

plt.legend(loc='upper right')

plt.xlabel("Epoch")
plt.ylabel("Error")

plt.show()

# Validate the model on the test data.

T1_output = sess.run(tf.nn.softmax(T1_layer), {X: liverpool_test})
T2_output = sess.run(tf.nn.softmax(T2_layer), {X: schalke_test})
T3_output = sess.run(tf.nn.softmax(T3_layer), {X: psg_test})

# Discretize the outputs so that the outcome is either Win, Tie, or Loss.
T1_output = (T1_output == T1_output.max(axis=1)[:,None]).astype(int)
T2_output = (T2_output == T2_output.max(axis=1)[:,None]).astype(int)
T3_output = (T3_output == T3_output.max(axis=1)[:,None]).astype(int)

T1_accuracy = accuracy_score(T1_output, liverpool_test_labels)
T2_accuracy = accuracy_score(T2_output, schalke_test_labels)
T3_accuracy = accuracy_score(T3_output, psg_test_labels)

print("T1 Accuracy: " + str(T1_accuracy))
print("T2 Accuracy: " + str(T2_accuracy))
print("T3 Accuracy: " + str(T3_accuracy))
