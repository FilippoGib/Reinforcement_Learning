import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np


# Define the epsilon-greedy policy
def next_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

def sarsa(env, num_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma):
    # First initialize Q with zeros
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    reward_per_episode = []

    # Enable interactive mode for live plotting
    plt.ion()
    demo_fig = plt.figure("Live Demo")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        action = next_action(Q, state, epsilon, n_actions)
        total_reward = 0

        while not done:
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            next_action_ = next_action(Q, next_state, epsilon, n_actions)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action_] - Q[state, action])
            state = next_state
            action = next_action_

        reward_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(reward_per_episode[-500:])
            print(f"Episode {episode+1}: Avg Reward {avg_reward:.2f}")

            # Run a full demonstration episode using the greedy policy (no exploration)
            demo_state, _ = env.reset()
            demo_done = False
            while not demo_done:
                frame = env.render()  # Get the current frame (assumes render_mode supports this)
                plt.figure(demo_fig.number)
                plt.clf()
                plt.imshow(frame)
                plt.title(f"Live Demo at Episode {episode+1}")
                plt.draw()
                plt.pause(.2)  # .2 second delay per action

                demo_action = next_action(Q, state=demo_state, epsilon=epsilon, n_actions=n_actions)  # e-greedy action from current Q
                demo_state, reward, demo_done, truncated, info = env.step(demo_action)
    plt.close()
    env.close()
    return Q, reward_per_episode



# Courtesy of chatgpt
def test_agent(Q, episodes=1):
    """Runs the trained agent for a few episodes to see its performance."""
    env = gym.make("FrozenLake-v1", is_slippery = False, render_mode="rgb_array")  # Use RGB mode for visualization

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        print(f"\nFinal test {episode+1}:")

        while not done:
            action = np.argmax(Q[state])  # Always take the best action
            state, reward, done, truncated, info = env.step(action)

            # Render the current frame
            frame = env.render()
            plt.imshow(frame)
            plt.title(f"Final test {episode+1}")
            plt.show()  # Show image without blocking execution
            plt.pause(1.0)  # Pause for half a second to observe

            if done:
                if reward > 0:
                    print("🏆 Success!")
                else:
                    print("❌ Failed.")
                break
        plt.close()
    env.close()



def main():
    env = gym.make("FrozenLake-v1", is_slippery = False, render_mode="rgb_array") # is_slippery = False means the environment is deterministic
    num_episodes = 5_000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    alpha = 0.1
    gamma = 0.90
    Q, rewards_per_episode = sarsa(env, num_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma)
    env.close()
    test_agent(Q)
    
    # Plot the learning progress
    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid'))  # Smooth trend
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward (100-episode window)")
    ax.set_title("SARSA Learning Progress")
    plt.show()

    policy = np.argmax(Q, axis=1)
    print(policy)

    # Visualize the learned policy as arrows on a grid
    policy = policy.reshape(4, 4)
    # Map actions to arrow symbols
    arrow_map = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    
    # Define terminal states for FrozenLake-v1
    terminal_states = [5, 7, 11, 12, 15]

    fig, ax = plt.subplots(figsize=(4, 4))
    # Create a grid layout
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.grid(True)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()  # So the top row corresponds to row 0

    # Annotate each cell with the corresponding arrow or colored box for terminal states
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            if state == 15:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='yellow'))
            elif state in terminal_states:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='blue'))
            else:
                action = policy[i, j]
                ax.text(j + 0.5, i + 0.5, arrow_map[action], 
                        ha='center', va='center', fontsize=20)
            if state == 0:
                # Add transparent green box for the starting point
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))

    plt.title("Learned Policy")
    plt.show()

    input()
if __name__ == "__main__":
    main()

