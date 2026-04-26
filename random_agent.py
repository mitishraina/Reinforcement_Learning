import gymnasium as gym
import numpy as np
import cv2

# cliffEnv = gym.make("CliffWalking-v1", render_mode="human")
cliffEnv = gym.make("CliffWalking-v1")
# cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")

def initalize_world():
    width, height = 600, 300
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2
    
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    # Start
    # frame = cv2.putText(img, text="S", org=(49 * 0 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame

    
    
def put_agent(img, obs):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=obs, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img

obs, info = cliffEnv.reset()
done = False
frame = initalize_world()
while not done:
    # print(cliffEnv.render())
    frame2 = put_agent(frame.copy(), obs)
    cv2.imshow("cliff world", frame2)
    cv2.waitKey(250)
    action = cliffEnv.action_space.sample()
    print(f"action taken---> [{['up', 'right', 'down', 'left'][action]}]")
    obs, reward, terminatad, truncated, info = cliffEnv.step(action)
    done = terminatad or truncated
cliffEnv.close()