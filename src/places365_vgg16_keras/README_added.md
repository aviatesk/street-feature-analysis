# Modifications added by aviatesk

Note that this directory is originally cloned from [Keras | VGG16 Places365 - VGG16 CNN models pre-trained on Places365-Standard for scene classification](https://github.com/GKalliatakis/Keras-VGG16-places365) by [GKalliatakis](https://github.com/GKalliatakis).

I removed the original Git files and  added some modifications to source code ([vgg16_hybrid_places_1365.py](./vgg16_hybrid_places_1365.py#L26), [vgg16_places_365.py](./vgg16_places_365.py#L23)) so that they works in my environment.

- Before: `from keras.applications.imagenet_utils import _obtain_input_shape`
- After: `from keras_applications.imagenet_utils import _obtain_input_shape`,



## Author of this README_added.md

- **KADOWAKI, Shuhei** - *Undergraduate@Kyoto Univ.* - [aviatesk]
