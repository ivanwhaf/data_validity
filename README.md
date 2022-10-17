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
