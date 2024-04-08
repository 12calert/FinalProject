from model1 import Model1
from model2 import Model2
import ganmodel
import classifier2
import classifier3

n_domains = 20
model1 = Model1()
model1.load()
model1_domains = model1.generate(n_domains)

model2 = Model1()
model2.load()
model2_domains = model2.generate(n_domains)

ganmodel.load()
ganmodel_domains = ganmodel.generate(n_domains)

classifier2.load()
predictions1 = classifier2.predict(model1_domains)
predictions2 = classifier2.predict(model2_domains)
predictions3 = classifier2.predict(ganmodel_domains)

classifier3.load()
predictions4 = classifier3.predict(model1_domains)
predictions5 = classifier3.predict(model2_domains)
predictions6 = classifier3.predict(ganmodel_domains)

def score(predictions):
    model_score = 0
    for p in predictions:
        if p[0] > 0.5:
            model_score += 1
    return model_score  / len(predictions)

print(predictions1)
print(predictions2)
print(predictions3)

print(score(predictions1))
print(score(predictions2))
print(score(predictions3))


print(predictions4)
print(predictions5)
print(predictions6)

print(score(predictions4))
print(score(predictions5))
print(score(predictions6))