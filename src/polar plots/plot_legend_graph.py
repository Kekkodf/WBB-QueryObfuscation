import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)

def main():
    # Plot a polar plot
    word = 'V'
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rmin(0)
    ax.set_rmax(20)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rmin(0)
    ax.set_rmax(10)

    ax.scatter(120, 6, marker='o', s=100, color='grey', alpha=0.2, label='word')
    ax.scatter(240, 3, marker='x', s=100, label='synonym')
    ax.scatter(360, 8, marker='s', s=100, label='hyponym')

    # Set r label as 'Euclidean distance'
    #ax.set_ylabel('Euclidean distance', fontsize=14)

    # Set theta label as 'α angle'
    #ax.set_xlabel('$\\alpha$ angle', fontsize=14)

    #ax.set_title('Word: %s' % word, fontsize=20)
    ax.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1)
    plt.show()

    return None

if __name__ == '__main__':
    main()
