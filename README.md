# EchoState_Network


EchoState_Network

### Install

```gitexclude
git clone https://github.com/rajboopathiking/EchoState_Network.git
```
```sh
cd EchoState_Network
```

### install ripser

```shell
pip install ripser
```
### TDA Feature Extraction 
```python

from ripser import ripser


tda_features_train = []
tda_features_test = []

for i in tqdm(range(X_train.shape[0])):
    diagram = ripser(X_train[i].reshape(-1, 1))['dgms']
    tda_features_train.append(np.mean([d[1] - d[0] for d in diagram[0] if d[1] != np.inf]))

for i in tqdm(range(X_test.shape[0])):
    diagram = ripser(X_test[i].reshape(-1, 1))['dgms']
    tda_features_test.append(np.mean([d[1] - d[0] for d in diagram[0] if d[1] != np.inf]))

tda_features_train = np.array(tda_features_train).reshape(-1, 1)
tda_features_test = np.array(tda_features_test).reshape(-1, 1)

```
### Echo State Network 
```python

import EchoState_NN
esn = EchoState_NN.EchoStateNetwork(
    n_inputs=X_train.shape[0],
    n_reservoir=100,
    n_outputs=1,
    spectral_radius=0.95,
    input_scaling=1.0,
    leakage_rate=1.0,
    random_state=42
)
```
### Create Signature Using Echo State

```python
# Collect reservoir states for the train and test sets
reservoir_states_train = []
reservoir_states_test = []

for i in tqdm(range(X_train.shape[0])):
    esn.state = esn._update_state(X_train[i])
    reservoir_states_train.append(esn.state)

for i in tqdm(range(X_test.shape[0])):
    esn.state = esn._update_state(X_test[i])
    reservoir_states_test.append(esn.state)

reservoir_states_train = np.array(reservoir_states_train)
reservoir_states_test = np.array(reservoir_states_test)

# Combine TDA features and reservoir states
signatures_train = np.hstack((tda_features_train, reservoir_states_train))
signatures_test = np.hstack((tda_features_test, reservoir_states_test))

print("Train Signatures Shape:", signatures_train.shape)
print("Test Signatures Shape:", signatures_test.shape)

```