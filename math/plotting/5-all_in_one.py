#!/usr/bin/bash python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 2])

# data one
y0 = np.arange(0, 11) ** 3
x0 = np.arange(0, 11)

# data two
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

# data three
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2/t2)*x2)

# data four
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32)*x3)

# data five
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# plot 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x0, y0, color='red')
ax1.set_xlim(left=0, right=10)


# plot 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x1, y1, color='magenta', s=5)
ax2.set_xlabel('Height (in)')
ax2.set_ylabel('Weight (lbs)')
ax2.set_title('Men\'s Height vs Weight')
ax2.set_xticks([60, 70, 80])

# plot 3
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x2, y2)
ax3.set_title('Exponential Decay of C-14')
ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Fraction Remaining')
ax3.set_xlim(left=0, right=28650)
ax3.set_yscale('log')
ax3.set_xticks([10000, 20000])

# plot 4
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x3, y31, color='red', linestyle='--')
ax4.plot(x3, y32, color='green')

ax4.set_title('Exponential Decay of Radioactive Elements')
ax4.set_xlabel('Time (years)')
ax4.set_ylabel('Fraction Remaining')
ax4.legend(['C-14', 'Ra-226'])
ax4.set_xlim(left=0, right=20000)
ax4.set_ylim(bottom=0, top=1)
ax4.set_xticks(range(0, 20001, 5000))
# ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax4.set_yticks([0, 0.5, 1])

# plot 5
ax5 = fig.add_subplot(gs[2, :])
ax5.hist(student_grades, edgecolor='black', bins=range(0, 101, 10))

ax5.set_title('Project A')
ax5.set_xlabel('Grades')
ax5.set_ylabel('Number of Students')
ax5.set_xlim(left=0, right=100)
ax5.set_xticks(range(0, 101, 10))
ax5.set_ylim(bottom=0, top=30)

fig.suptitle('All in One')
plt.tight_layout()

plt.show()
