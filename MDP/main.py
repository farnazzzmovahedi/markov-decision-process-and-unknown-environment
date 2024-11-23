import pygame
from environment import PygameInit, AngryBirds, value_iteration

if __name__ == "__main__":
    FPS = 64
    env = AngryBirds()
    screen, clock = PygameInit.initialization()
    state = env.reset()
    sum_rewards = 0

    # Compute the optimal policy using combined value iteration
    policy = value_iteration(env.grid, env.transition_table)

    # Print the policy
    action_labels = {
        0: "↑",  # Up
        1: "↓",  # Down
        2: "←",  # Left
        3: "→"   # Right
    }

    print("\nOptimal Policy:")
    for x in range(8):
        row = []
        for y in range(8):
            if env.grid[x][y] == "R":
                row.append("R")  # Rock
            else:
                action = policy[x, y]
                row.append(action_labels[action] if action != -1 else " ")
        print(" ".join(row))

    # Run the game simulation
    for _ in range(5):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)

            # Extract action from policy
            row, col = state  # Unpack the current state
            action = policy[row, col]  # Access policy using indices
            next_state, probability, reward_episode, done = env.step(action)
            # print(reward_episode)
            sum_rewards += reward_episode

            if done:
                print(f"Episode finished with reward: {sum_rewards}")
                sum_rewards = 0
                state = env.reset()
            else:
                state = next_state  # Update state to the next state

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()





