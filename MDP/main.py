import pygame

from environment import PygameInit, AngryBirds, value_iteration, print_policy, plot_delta_convergence, print_state_value
import matplotlib
matplotlib.use("Agg")  # Use a backend that doesn't require Tk

if __name__ == "__main__":

    FPS = 10
    env = AngryBirds()
    screen, clock = PygameInit.initialization()  # Initialize Pygame (commented out)
    state = env.reset()
    sum_rewards = 0
    sum_episodes_reward = 0
    temp = 0
    agent_poos = env.get_agent_position()
    policy = None

    # Initialize variables to track the nearest pig
    nearest_pig = None

    goal_policy, V, delta_history = value_iteration(env, env.transition_table, phase='goal')
    # Visualize the convergence of delta
    # plot_delta_convergence(delta_history)
    print_state_value(V)

    print_policy(goal_policy, env.grid)

    # print the grid
    for row in env.grid:
        print(" ".join(row))

    for episode in range(5):
        running = True
        while running:

            # Comment out Pygame event handling and rendering
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    exit()

            env.render(screen)  # Comment out rendering

            if policy is None:
                policy, V, _ = value_iteration(env, env.transition_table, phase='pigs')

            while env.pigs_eaten < 7:
                if env.pigs_eaten != temp:
                    policy, V, _ = value_iteration(env, env.transition_table, phase='pigs')  # Recompute policy for pigs phase

                    temp = env.pigs_eaten

                # Extract action from policy
                row, col = state  # Unpack the current state
                action = policy.get((row, col), 0)  # Access policy, default to 0 (Up) if state is not in policy
                next_state, probability, reward_episode, done = env.step(action)
                sum_rewards += reward_episode

                if done:
                    print(f"Episode finished with total reward: {sum_rewards}")
                    sum_episodes_reward = sum_episodes_reward + sum_rewards
                    if episode == 4:
                        print(f"Avg rewards: {sum_episodes_reward / 5}")
                    sum_rewards = 0
                    state = env.reset()  # Reset environment for the next episode
                    running = False  # Stop the current episode loop
                    policy = None
                else:
                    state = next_state

            policy = goal_policy

            # Extract action from policy
            row, col = state  # Unpack the current state
            action = policy.get((row, col), 0)  # Access policy, default to 0 (Up) if state is not in policy
            next_state, probability, reward_episode, done = env.step(action)
            sum_rewards += reward_episode

            if done:
                print(f"Episode finished with total reward: {sum_rewards}")
                sum_episodes_reward = sum_episodes_reward + sum_rewards
                if episode == 4:
                    print(f"Avg rewards: {sum_episodes_reward / 5}")
                sum_rewards = 0
                state = env.reset()  # Reset environment for the next episode
                running = False  # Stop the current episode loop
                policy = None
            else:
                state = next_state  # Update state to the next state

            pygame.display.flip()  # Comment out Pygame display update
            clock.tick(FPS)  # Comment out Pygame clock tick

    pygame.quit()  # Comment out Pygame quit
