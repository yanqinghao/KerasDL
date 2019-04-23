import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port.
ptvsd.enable_attach(address=('10.122.24.46', 3000), redirect_output=True)

# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models

model = load_model('cats_and_dogs_small_2.h5')
print(model.summary())

img_path = './data/cats_and_dogs_small/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

# plt.imshow(img_tensor[0])
# plt.show()
plt.imsave('./result/dog_vs_cat_visual_1_cat.png', img_tensor[0])

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.imsave(
    './result/dog_vs_cat_visual_1_4chn.png',
    first_layer_activation[0, :, :, 4],
    cmap='viridis')
# plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.imsave(
    './result/dog_vs_cat_visual_1_7chn.png',
    first_layer_activation[0, :, :, 7],
    cmap='viridis')

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row +
                                             row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size:(col + 1) * size, row * size:(row + 1) *
                         size] = channel_image
    scale = 1. / size
    plt.figure(
        figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imsave(
        './result/dog_vs_cat_visual_1_' + layer_name + '.png',
        display_grid,
        cmap='viridis')
