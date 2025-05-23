import pygame

def play_human(env, i=1):
    for j in range(i):
        obs, info =env.reset()
    #obs, info = env.reset()
    running = True
    env.render()

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return obs, reward, info
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = 0
                elif event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
        if action is not None and action in info["legal_actions"]:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
            if done:
                running = False
                print("Episode finished")
                return obs, reward, info
            
