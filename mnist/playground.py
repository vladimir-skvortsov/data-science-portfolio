import pygame
import pygame.locals as locals
from tensorflow import keras
import numpy as np
import cv2
import math

model = keras.saving.load_model('mnist.h5')

pygame.init()

screen = pygame.display.set_mode((512, 512))
screen.fill((0, 0, 0))

running = True
drawing = False
last_pos = None

pad = 5

while running:
  for event in pygame.event.get():
    if event.type == locals.QUIT:
      running = False
    if event.type == locals.MOUSEBUTTONDOWN:
      drawing = True
    if event.type == locals.MOUSEBUTTONUP:
      drawing = False
      last_pos = None
    if event.type == locals.KEYDOWN and event.key == locals.K_SPACE:
      screen.fill((0, 0, 0))
    if event.type == locals.MOUSEMOTION:
      if drawing:
        pos = pygame.mouse.get_pos()

        pygame.draw.circle(screen, (255, 255, 255), pos, 10)

        if last_pos:
          pygame.draw.line(screen, (255, 255, 255), last_pos, pos, 22)

        last_pos = pos

  image = pygame.surfarray.array3d(screen)
  image = np.rot90(image)
  image = np.flipud(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image = image / 255.0
  coords = cv2.findNonZero(image)
  x, y, w, h = cv2.boundingRect(coords)
  if (w != 0 and h != 0):
    image = image[y:y+h, x:x+w]

    n = max(w, h)
    pad_left = math.ceil((n - w) / 2) + 6
    pad_right = math.floor((n - w) / 2) + 6
    pad_top = math.ceil((n - h) / 2) + 5
    pad_bottom = math.floor((n - h) / 2) + 3
    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')

    image = cv2.resize(image, (28, 28))
    image = image.reshape(28, 28, 1)
    mean_px = image.mean()
    std_px = image.std()
    image = (image - mean_px) / std_px

    pred = model.predict(np.array([image]), verbose=0)[0]
    digit = np.argmax(pred)
    percent = pred[digit] * 100

    pygame.display.set_caption(f'Prediction: {digit} ({percent:.2f}%)')

  pygame.display.flip()
