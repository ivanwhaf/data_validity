# data_validity

Global data validity tools: including 20+ data data validity evaluation algorithms

here are some demos:


## Forgetting Events

### Example

```python
from forgetting_events import ForgettingEvents
from model import ResNet50

train_set = MNIST(root, train=True)
train_loader = DataLoader(train_set, batch_size=64, huffle=True)
model = ResNet50()

fe = ForgettingEvents(len(train_set))  # forgetting events!

for epoch in range(100):
    for batch_idx, (inputs, labels, indices) in enumerate(train_loader):
        outputs = model(inputs)
        pred_labels = outputs.max(1, keepdim=True)[1]

        fe.record_forgetting(pred_labels, labels, indices)  # record forgetting events!

        # update network
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

scores = fe.forgetting_times  # data validity scores

# select high-score samples for retraining
```

## EMA

### Example

A naive training loop:

```python
from ema import EMA
from model import ResNet50

model = ResNet50()  # your own network
ema = EMA(model, beta=0.85)  # ema model!

for epoch in range(100):
    train(model, train_loader, optimizer, criterion)

    ema.update()  # update mean teacher model(EMA) after backward params
    ema.apply_teacher()  # use mean teacher model(EMA) before evaluating

    validate(model, val_loader)

    ema.restore_student()  # restore to student model(raw model) after evaluating

ema.apply_teacher()  # use mean teacher model(EMA) before testing
test(mode, test_loader)

save_mode(model)  # save ema model
ema.restore_student()
save_mode(model)  # save raw model
```

* you can see `demo/train_ema.py` for more detail

## Online Ensemble

### Example

#### Step1: Pretrain ensemble models

* split offline clean dataset into different ratio part:

```python
sub_set1, sub_set2, sub_set3 = torch.utils.data.random_split(train_set, [10000, 20000, 50000])
train_loader1 = DataLoader(sub_set1, batch_size=64, shuffle=True)
train_loader2 = DataLoader(sub_set2, batch_size=64, shuffle=True)
train_loader3 = DataLoader(sub_set3, batch_size=64, shuffle=True)
```

* pretrain base models:

```python
train_loaders = [train_loader1, train_loader2, train_loader3]
for i, train_loader in enumerate(train_loaders):
    model = ResNet50()  # your own model
    for epoch in range(100):
        train(model, train_loader, criterion)
    torch.save(model.state_dict(), './model' + str(i) + '.pth')  # save each ensemble model
```

#### Step2: Load ensemble models

```python
from online_ensemble import OnlineEnsemble

oe = OnlineEnsemble()
models = []
model_path = ['./model0.pth', './model1.pth', './model2.pth']
for path in model_path:
    model = ResNet50()
    model.load_state_dict(torch.load(path))  # load pretrain base models
    models.append(model)

oe.load_pretrain_ensemble_models(models)

```

#### Step3: Inference online data batch and record scores

```python
batch_dataset = MNIST(root, train=True)  # batch dataset
batch_data_loader = DataLoader(batch_dataset, batch_size=64, huffle=True)
scores = np.zeros(len(batch_dataset))

for idx, (inputs, labels, indices) in enumerate(batch_data_loader):
    for model in oe.ensemble_models:
        outputs = model(inputs) # current data batch
        loss = criterion(outputs, labels)
        loss = loss.detach().cpu().numpy()
        scores[indices] += loss  # here use loss as score, you can convert to any task-based quota!
```

#### Step4: Sort online data batch by scores and Resample / Reweight, then Retrain

* sort online data batch and resample + retrain

```python
indices = oe.sort_and_resample(scores, len(batch_dataset), 0.05, 0.95)  # resample
new_batch_dataset = Subset(batch_dataset, indices)
new_train_loader = DataLoader(new_batch_dataset, batch_size=64, huffle=True)

for idx, (inputs, labels, indices) in enumerate(new_train_loader):
    # update network using new data batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

* sort online data batch and reweight + retrain

```python
weights = oe.sort_and_reweight(scores, len(batch_dataset), 0.05, 0.95)  # reweight
batch_train_loader = DataLoader(batch_dataset, batch_size=64, huffle=True)

for idx, (inputs, labels, indices) in enumerate(batch_train_loader):
    # update network using new weights
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    weight = weights[indices]
    loss = loss * weight  # multiply weights！
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

* you can see `demo/pretrain_ensemble_models.py` for more detail



* you can see `demo/train_forgetting_events.py`