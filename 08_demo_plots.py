import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd


# demo question 1
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')  # Hide axes

# Draw the question box
question_box = mpatches.Rectangle((0, 6), 10, 1, linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(question_box)
ax.text(0.2, 6.5, "Question: Denali and Nate work for a dog walking business and are paid for ...", fontsize=12, fontweight='bold', verticalalignment='center')

response_text = [
    ("Response:", 'bold', 'black'),
    ("To solve this problem, let's start by defining the variables and setting up the equations.", 'normal', 'black'),
    ("based on the given information.", 'normal', 'black'),
    (r"1. Let $D$ be the number of dogs Denali walks.", 'normal', 'black'),
    (r"2. Let $N$ be the number of dogs Nate walks.", 'normal', 'black'),
    (r"3. Initially, $D = 16$ and $N = 12$.", 'normal', 'black'),
    ("According to the problem, the ratio of Denali's pay to Nate's pay would be the same if ...", 'normal', 'black'),
    ("...", 'normal', 'black'),
    (r"First, we solve $\frac{16+4x}{12} = \frac{4}{3}$:", 'normal', 'black'),
    (r"$16 + 4x = \frac{4 \times 12}{3} = 16 \Rightarrow 4x = 0$", 'normal', 'black'),
    (r"$\Rightarrow x = 0$, but since $x \neq 0$, this equation doesn't provide a valid solution.", 'normal', 'black'),
    ("Let's check if we made an error. We should verify the problem conditions again.", 'bold', 'red'),
    ("The correct approach is to directly solve the ratio equality...", 'normal', 'black'),
    ("...", 'normal', 'black'),
    (r"Since $x \neq 0$, the solution is $x = 5$.", 'normal', 'black'),
    (r"Therefore, the value of $x$ is $5$", 'normal', 'black'),
]
y_pos = 5.5
for text, weight, color in response_text:
    ax.text(0.2, y_pos, text, fontsize=12, fontweight=weight, color=color, verticalalignment='center')
    y_pos -= 0.5  # Move down for the next line
plt.savefig("responses/plot_demo_question_1.png", dpi=300, bbox_inches='tight')
plt.savefig("responses/plot_demo_question_1.svg", format='svg', dpi=300, bbox_inches='tight')


# demo question 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.axis('off')  # Hide axes

question_box = mpatches.Rectangle((0, 6), 8, 1, linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(question_box)
ax.text(0.2, 6.5, "Question: The sum of all the positive factors of integer $x$ is 24.\nIf one of the factors is 3, what is the value of x?", fontsize=12, fontweight='bold', verticalalignment='center', wrap=True)

response_text = [
    ("Response:", 'bold', 'black'),
    (r"To solve this problem, we need to find the integer $x$ such that the sum", 'normal', 'black'),
    ("of all its positive factors is 24 and one of its factors is 3.", 'normal', 'black'),
    ("Let's break down the steps:", 'normal', 'black'),
    (r"1. Identify the factors of $x$.", 'normal', 'black'),
    ("2. Calculate the sum of these factors.", 'normal', 'black'),
    ("3. Check if the sum is 24 and if 3 is one of the factors.", 'normal', 'black'),
    ("...", 'normal', 'black'),
    ("It seems that the initial approach did not yield the correct result.", 'normal', 'black'),
    ("Let's rethink the problem and simplify our approach.", 'bold', 'red'),
    (r"Since $x$ is divisible by 3, let's consider the possible values...", 'normal', 'black'),
    ("...", 'normal', 'black'),
    (r"The value of $x$ that satisfies the conditions of the problem is 15", 'normal', 'black'),
]
# Plot the response text line by line
y_pos = 5.5
for text, weight, color in response_text:
    ax.text(0.2, y_pos, text, fontsize=12, fontweight=weight, color=color, verticalalignment='center')
    y_pos -= 0.5  # Move down for the next line
plt.savefig("responses/plot_demo_question_2.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_demo_question_2.svg", format='svg', dpi=300, bbox_inches='tight')



# type-1: double check and correct
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.axis('off')  # Hide axes

# Draw question box
question_box = mpatches.Rectangle((0, 6), 8, 1, linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(question_box)
ax.text(0.2, 6.7, r"Question: The median of the set $\left\{n, n+5, n+6, n+9, n+15\right\}$ is 9.", fontsize=12, fontweight='bold', verticalalignment='center', ha='left')
ax.text(0.2, 6.25, r"What is the mean?", fontsize=12, fontweight='bold', verticalalignment='center', ha='left')
response_text = [
    ("Response:", 'bold', 'black'),
    (r"To solve this problem, we need to determine the value of $n$ using the", 'normal', 'black'),
    ("information about the median, and then use it to find the mean of the set.", 'normal', 'black'),
    ("1. Identify the median: ...", 'normal', 'black'),
    (r"2. Determine the set with the value of $n$: ...", 'normal', 'black'),
    ("3. Calculate the mean: ...", 'normal', 'black'),
    ("Let's verify this with Python code to ensure accuracy.", 'normal', 'black'),
    (r"Calculate the mean, the output $10.0$", 'normal', 'black'),
    ("Let's recheck the steps:", 'bold', 'red'),
    (r"The median of the set $\left\{n, n + 5, n + 6, n + 9, n + 15\right\}$ is $(n + 6)$.", 'normal', 'black'),
    ("...", 'normal', 'black'),
    (r"Therefore, the mean of the set is $10$.", 'normal', 'black'),
]
y_pos = 5.5
for text, weight, color in response_text:
    ax.text(0.2, y_pos, text, fontsize=12, fontweight=weight, color=color, verticalalignment='center', ha='left')
    y_pos -= 0.5  # Move down for the next line
plt.savefig("responses/plot_demo_type1.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_demo_type1.svg", format='svg', dpi=300, bbox_inches='tight')


# type-2: self-reflection leads to corerct answer
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.axis('off')  # Hide axes

# Draw question box
question_box = mpatches.Rectangle((0, 6), 8, 1, linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(question_box)
ax.text(0.2, 6.7, r"Question: Krzysztof solved the quadratic equation $11x^2-44x-99=0$", fontsize=11, fontweight='bold', verticalalignment='center', ha='left')
ax.text(0.2, 6.25, r"by completing the square. In the process, ... What is $r+s$?", fontsize=11, fontweight='bold', verticalalignment='center', ha='left')
response_text = [
    ("Response:", 'bold', 'black'),
    (r"To solve the quadratic equation $(11x^2 - 44x - 99 = 0)$ by completing the square,", 'normal', 'black'),
    ("we can follow these steps:", 'normal', 'black'),
    ("1. Divide the entire equation by 11 to simplify it", 'normal', 'black'),
    ("...", 'normal', 'black'),
    (r"output $15.0$", 'normal', 'black'),
    ("It appears there was a mistake in the manual calculation.", 'bold', 'red'),
    ("Let's verify this with Python code to ensure accuracy.", 'bold', 'red'),
    (r"1. Start with the simplified equation $(x^2 - 4x - 9 = 0)$.", 'normal', 'black'),
    ("...", 'normal', 'black'),
    ("4. This gives us ((x - 2)^2 = 13", 'normal', 'black'),
    (r"The median of the set $\left\{n, n + 5, n + 6, n + 9, n + 15\right\}$ is $(n + 6)$.", 'normal', 'black'),
    ("...", 'normal', 'black'),
    (r"So, $(r = -2)$ and $(s = 13)$, and $(r + s = -2 + 13 = 11)$.", 'normal', 'black'),
    (r"The correct answer should be $11$.", 'normal', 'black'),
]
y_pos = 5.5
for text, weight, color in response_text:
    ax.text(0.2, y_pos, text, fontsize=11, fontweight=weight, color=color, verticalalignment='center', ha='left')
    y_pos -= 0.5  # Move down for the next line
plt.savefig("responses/plot_demo_type2.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_demo_type2.svg", format='svg', dpi=300, bbox_inches='tight')



# type-3: self-reflection introduces error
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.axis('off')  # Hide axes

# Draw question box
question_box = mpatches.Rectangle((0, 6), 8, 1, linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(question_box)
ax.text(0.2, 6.5, r"Question: Solve for $x$: $81^{2x} = 27^{3x - 4}$.", fontsize=12, fontweight='bold', verticalalignment='center', ha='left')
response_text = [
    ("Response:", 'bold', 'black'),
    ("First, express both sides of the equation with the same base.", 'normal', 'black'),
    ("One of its factors is 3, and the sum of all its positive factors is 24.", 'normal', 'black'),
    ("It appears there was an error in the verification of the second equation.", 'normal', 'black'),
    ("Let's recheck the calculations manually and ensure the code", 'bold', 'red'),
    ("correctly reflects the mathematical steps.", 'bold', 'red'),
    (r"For the first equation $81^{2x} = 27^{3x - 4}$, we verified that $x = 12$ is correct.", 'normal', 'black'),
    ("...", 'normal', 'black'),
    ("Let's recheck the code with a more precise approach.", 'bold', 'red'),
    (r"Rewriting with the same base: $(3^{-1})^{2x + 1} = (3^{-3})^{x - 1} $", 'normal', 'black'),
    ("...", 'normal', 'black'),
    ("Let's rethink the problem and simplify our approach.", 'bold', 'red'),
    (r"For the second equation $\left(\frac{1}{3}\right)^{2x + 1} = \left(\frac{1}{27}\right)^{x - 1}$", 'normal', 'black'),
    ("...", 'normal', 'black'),
    ("The recheck confirms that $x = 4$ is indeed the correct solution", 'normal', 'black'),
    ("for the second equation.", 'normal', 'black'),
    (r"The final answers are: $ 12 \quad \text{and} \quad 4$", 'normal', 'black'),
]
y_pos = 5.5
for text, weight, color in response_text:
    ax.text(0.2, y_pos, text, fontsize=12, fontweight=weight, color=color, verticalalignment='center', ha='left')
    y_pos -= 0.5  # Move down for the next line
plt.savefig("responses/plot_demo_type3.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_demo_type3.svg", format='svg', dpi=300, bbox_inches='tight')


# type-4: repetitive self-reflections
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.axis('off')  # Hide axes

# Draw question box
question_box = mpatches.Rectangle((0, 6), 8, 1, linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(question_box)
ax.text(0.2, 6.5, r"In triangle $ABC$, $AB = BC$, and $\overline{BD}$ is an altitude...", fontsize=12, fontweight='bold', verticalalignment='center', ha='left')
response_text = [
    ("Response:", 'bold', 'black'),
    ("To solve the problem, let's define the necessary variables and.", 'normal', 'black'),
    (r"use the given conditions to find the area of triangle $ABC$.", 'normal', 'black'),
    ("1. Define Variables", 'normal', 'black'),
    ("...", 'normal', 'black'),
    ("This is not possible, so let's recheck the conditions.", 'bold', 'red'),
    (r"We should have: $2 \left(\frac{10 + x}{h}\right) = \frac{x + 10}{h}$", 'normal', 'black'),
    (r"$\Rightarrow 20 + 2x = x + 10 \Rightarrow x = -10 \Rightarrow ... \Rightarrow h=0$", 'normal', 'black'),
    ("This is not possible, so let's recheck the conditions.", 'bold', 'red'),
    (r"We should have: $2 \left(\frac{10 + x}{h}\right) = \frac{x + 10}{h}$", 'normal', 'black'),
    (r"$\Rightarrow 20 + 2x = x + 10 \Rightarrow x = -10 \Rightarrow ... \Rightarrow h=0$", 'normal', 'black'),
    ("This is not possible, so let's recheck the conditions.", 'bold', 'red'),
    (r"We should have: $2 \left(\frac{10 + x}{h}\right) = \frac{x + 10}{h}$", 'normal', 'black'),
    (r"$\Rightarrow 20 + 2x = x + 10 \Rightarrow x = -10 \Rightarrow ... \Rightarrow h=0$", 'normal', 'black'),
    ("This is not possible, so let's recheck the conditions.", 'bold', 'red'),
    (r"We should have: $2 \left(\frac{10 + x}{h}\right) = \frac{x + 10}{h}$", 'normal', 'black'),
    (r"$\Rightarrow 20 + 2x = x + 10 \Rightarrow x = -10 \Rightarrow ... \Rightarrow h=0$", 'normal', 'black'),
    ("...", 'normal', 'black'),
]
y_pos = 5.5
for text, weight, color in response_text:
    ax.text(0.2, y_pos, text, fontsize=12, fontweight=weight, color=color, verticalalignment='center', ha='left')
    y_pos -= 0.5  # Move down for the next line
plt.savefig("responses/plot_demo_type4.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_demo_type4.svg", format='svg', dpi=300, bbox_inches='tight')


