from random import choice
from numpy import array, random, dot

unit_step = lambda x: 0 if x < 0 else 1

train_data = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]

w = random.rand(3)
errors = []
learning_rate = 0.2
n = 100

for i in xrange(n):
    x, expected = choice(train_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += learning_rate * error * x

for x, _ in train_data:
    result = dot(w, x)
    print ("{}: {} -> {}".format(x[:2], result, unit_step(result)))

