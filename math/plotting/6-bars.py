#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
colors = {'Apples': 'red', 'Bananas': 'yellow',
          'Oranges': '#ff8000', 'Peaches': '#ffe5b4'}
people = ['Farrah', 'Fred', 'Felicia']
width = 0.5

x = np.arange(len(people))
fig, ax = plt.subplots()

# Initialize bottom for stacking
bottoms = np.zeros(len(people))

# Define the fruit types for the legend and colors
fruit_types = ['Apples', 'Bananas', 'Oranges', 'Peaches']

for i, fruit_type in enumerate(fruit_types):
    ax.bar(x, fruit[i], width, color=colors[fruit_type],
           label=fruit_type, bottom=bottoms)
    bottoms += fruit[i]  # Update the bottom position for the next stack

# Set the title, labels, and legend
ax.set_title('Number of Fruit per Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_xticks(x)
ax.set_xticklabels(people)
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

plt.show()
