import matplotlib.pyplot as plt

def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
