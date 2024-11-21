import pygame
from environment import PygameInit, AngryBirds

if __name__ == "__main__":

    FPS = 8
    env = AngryBirds()
    screen, clock = PygameInit.initialization()
    state = env.reset()

    for _ in range(5):
        running = True
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)

            # extract action from policy
            action = 0
            next_state, probability, reward_episode, done = env.step(action)

            if done:
                print(f"Episode finished with reward: {reward_episode}")
                state = env.reset()

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()


