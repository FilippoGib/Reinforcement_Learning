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

# Define the SARSA algorithm 
def sarsa_lambda(env, num_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, lambda_):
    # First inizialize Q with zeros
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    reward_per_episode = []

    for episode in range(num_episodes):
        # set state to inizial state
        state, _ = env.reset()
        done = False # The environment will tell me when I am done
        elegibility = np.zeros((n_states, n_actions)) # OSS elegibility referes to a couple (state, action)
        # pick the first action before the episode starts
        action = next_action(Q, state, epsilon, n_actions)
        total_reward = 0 # for statistics

        while not done:
            # actuate my action
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # choose next action from next_state before the update
            next_action_ = next_action(Q, next_state, epsilon, n_actions)
            # update Q
            delta = reward + gamma*Q[next_state][next_action_] - Q[state][action]
            elegibility[state][action] += 1
            Q[state][action] = Q[state, action] + alpha*delta*elegibility[state][action]
            elegibility = gamma*lambda_*elegibility # decay all the elegibility
            state = next_state
            action = next_action_

        reward_per_episode.append(total_reward) # save the total reward for this episode
        epsilon = max(epsilon_min, epsilon*epsilon_decay) # Update epsilon every episode

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(reward_per_episode[-500:])
            print(f"Episode {episode+1}: Avg Reward {avg_reward:.2f}")

    return Q, reward_per_episode



# Courtesy of chatgpt
def test_agent(Q, episodes=1):
    """Runs the trained agent for a few episodes to see its performance."""
    env = gym.make("FrozenLake-v1", is_slippery = False, render_mode="rgb_array")  # Use RGB mode for visualization

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        print(f"\nEpisode {episode+1}:")

        while not done:
            action = np.argmax(Q[state])  # Always take the best action
            state, reward, done, truncated, info = env.step(action)

            # Render the current frame
            frame = env.render()
            plt.imshow(frame)
            plt.title(f"Episode {episode+1}")
            plt.show(block=False)  # Show image without blocking execution
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
    lambda_ = 0.0
    Q, rewards_per_episode = sarsa_lambda(env, num_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, lambda_)
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



if __name__ == "__main__":
    main()


