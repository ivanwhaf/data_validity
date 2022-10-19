# data_validity

Data validation tools

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

* you can see `demo/train_ema_mnist.py` for more detail

## Online Ensemble

### Example

#### Step1: Pretrain ensemble models

```python
train_loaders = [train_loader1, trainloader2, trainloader3]
for i, train_loader in enumerate(train_loaders):
    model = ResNet50()  # your own model
    for epoch in range(100):
        train(model, train_loader, criterion)
    torch.save(model.state_dict(), './model' + str(i) + '.pth')  # save each ensemble model
```

#### Step2: Normal train loop, load ensemble models

```python
oe = OnlineEnsemble()
models = []
model_path = ['./model0.pth', './model1.pth', './model2.pth']
for path in model_path:
    model = ResNet50()  # your own model
    model.load_state_dict(torch.load(path))
    models.append(model)

oe.load_pretrain_ensemble_models(models)

```

#### Step3: Inference online data batch and record scores
* Inference dataloader shuffle must be False!!!
```python
dataset = MNIST(root, train=True)  # current data batch
inference_loader = DataLoader(dataset, batch_size=64, shuffle=False)  # inference dataloader shuffle must be False!!!!!!!!!!!

scores = np.zeros(len(dataset))
for idx, (inputs, labels) in enumerate(inference_loader):
    for model in oe.ensemble_models:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss.detach().cpu().numpy()
        scores[idx*batch_size:idx*batch_size+inputs.size(0)] += loss  # here use loss as score, you can convert to any task-based quota!
```

#### Step4: Sort online data batch by scores and Resample / Reweight -> Retrain
```python
outputs = oe.sort_and_resample(outputs, scores, outputs.size(0), 0.05, 0.95)  # get new outputs after resample
# update network using new data batch
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
or sort online data batch and reweight
```python
outputs, weights = oe.sort_and_reweight(outputs, scores, outputs.size(0), 0.05, 0.95)  # get new outputs after reweight
loss = criterion(outputs, labels)
loss = loss * weights  # multiply weights！
optimizer.zero_grad()
loss.backward()
optimizer.step()
```