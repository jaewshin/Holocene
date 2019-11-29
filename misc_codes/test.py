import numpy as np 
import matplotlib.pyplot as plt
def simulation_plots(L1, k1, x_start1, L2, k2, x_start2, figure):
    """
    Generate simulation plots for Figure 10 and 11
    """
    x = -np.random.chisquare(2,4000)
    x_lin = np.linspace(x.min(),x.max(),num=4000)

    y1 = L1/(1.+np.exp(-k1*(x-x_start1 )))
    y1_lin = L1/(1.+np.exp(-k1*(x_lin-x_start1 )))

    f, axes = plt.subplots(2, 2, figsize=(12,12))
    axes[0, 0].plot(x_lin, y1_lin)
    axes[0, 0].set_title("First CC growth")
    # plt.plot(x_lin,y1_lin)
    # plt.title('First CC growth')
    # # if figure==10:
    # #     plt.savefig('S1-a.png')
    # # else:
    # #     plt.savefig('S2-a.png')
    # plt.xlabel("time")
    # plt.ylabel("CC value")
    # plt.show()
    # plt.close() 

    y2 = L2/(1.+np.exp(-k2*(x-x_start2 )))
    y2_lin = L2/(1.+np.exp(-k2*(x_lin-x_start2 )))

    axes[0, 1].plot(x_lin, y2_lin)
    axes[0, 1].set_title("Second CC growth")
    # plt.plot(x_lin,y2_lin)
    # plt.title('Second CC growth')
    # # if figure==10:
    # #     plt.savefig('S1-b.png')
    # # else:
    # #     plt.savefig('S2-b.png')
    # plt.xlabel("time")
    # plt.ylabel("CC value")
    # plt.show()
    # plt.close()

    # n, bins, patches = plt.hist(y2, 200,  facecolor='blue', alpha=0.5)
    # plt.title('CC2')
    # plt.close()

    Y=.5*y1+.5*y2
    Y_lin=.5*y1_lin+.5*y2_lin
    axes[1,0].plot(x_lin, Y_lin)
    axes[1,0].set_title("Growth of the average")
    # plt.plot(x_lin,Y_lin)
    # plt.title('Growth of the average')
    # # if figure==10:
    # #     plt.savefig('S1-c.png')
    # # else:
    # #     plt.savefig('S2-c.png')
    # plt.xlabel("time")
    # plt.ylabel("CC value")
    # plt.show()

    # plt.close()

    n, bins, patches = plt.hist(Y, 200,  facecolor='blue', alpha=0.5)
    plt.title('Histogram of the average of two CC')
    axes[1, 1].hist(Y, 200, facecolor='blue', alpha=0.5)
    axes[1, 1].set_title('Histogram of the average of two CC')
    # plt.xlabel("time")
    # plt.ylabel("CC value")
    # plt.show()

    # plt.close()
    # f.set_xlabel("time")
    # f.set_ylabel("CC value")
    for ax in axes.flat:
        ax.set(xlabel='time', ylabel='CC value')
        # ax.set_fontsize(10)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.xaxis.get_label().set_fontsize(12)
        ax.yaxis.get_label().set_fontsize(12)
        ax.label_outer()

    if figure==10:
        plt.savefig('S1.png')
    else:
        plt.savefig('S2.png')
    plt.show()
    plt.clf()
    plt.close()

    print(f"Done with simulation models (Figure{figure})")

#Figure 10
simulation_plots(1, 1, -10, 5, 1, -1, 10)

#Figure 11
simulation_plots(1, 1, -10, 5, 1, -5, 11)
