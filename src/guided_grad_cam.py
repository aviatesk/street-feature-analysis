import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.framework import ops

H, W, C = 224, 224, 3
CITIES = ('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing', 'kyoto',
          'seoul', 'singapore', 'tokyo')
LABEL2CLASS1 = [(0, 'western'), (1, 'eastern')]
LABEL2CLASS2 = [(i, city_name) for i, city_name in enumerate(CITIES)]


def load_image(image_path, target_size=(H, W), preprocessing=False):
    '''
    load and preprocess an image from its path
    '''
    x = image.load_img(image_path, target_size=target_size)

    if preprocessing:

        x = image.img_to_array(x)[np.newaxis]
        gen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rescale=1. / 255,
        ).flow(
            x, batch_size=1, shuffle=False)
        x = next(gen)

    return x


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def l2_normalize(x):
    '''
    normalize a tensor by its L2 norm
    '''
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def set_guided_model(model_file_path):
    '''
    return a model, whose gradient function for all the ReLU activations according to guided backpropagation are changed
    '''

    if "GuidedBackProp" not in ops._gradient_registry._registry:

        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(
                op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = load_model(model_file_path)

    return new_model


def guided_backpropagation(guided_model, preprocessed_input, layer_name):
    '''
    Guided Backpropagation method for visualizing saliency of input image
    '''
    layer_output = guided_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, guided_model.input)[0]

    backprop_function = K.function(
        [guided_model.input, K.learning_phase()], [grads])
    grads_val = backprop_function([preprocessed_input, 0])[0]

    return grads_val


def compute_grad_cam(model,
                     preprocessed_input,
                     out,
                     cls,
                     layer_name,
                     output_shape=(W, H),
                     normalize=False):
    '''
    Grad-CAM method for visualizing saliency of input image

    argments:
    out: the kind or outputs (1: output1, 2: output2)
    output_shape: tuple of output image shape in (w, h) order
    '''
    y_c = model.output[out][0, cls]
    target_conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, target_conv_output)[0]

    if normalize:  # normalize if necessary
        grads = l2_normalize(grads)

    gradient_function = K.function([model.input], [target_conv_output, grads])
    output, grads_val = gradient_function([preprocessed_input])

    weights = np.mean(grads_val[0, :, :, :], axis=(0, 1))
    cam = np.dot(output[0, :], weights)

    # process CAM
    cam = cv2.resize(cam, output_shape, cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


def compute_grad_cam_batch(input_model, images, classes, layer_name):
    '''
    Grad-CAM method for visualizing input saliency, same as `compute_grad_cam` but process multiple images in one run
    '''

    loss = tf.gather_nd(input_model.output,
                        np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function(
        [input_model.input, K.learning_phase()], [layer_output, grads])

    target_conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', target_conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


def show_saliency(
        model,
        guided_model,
        image_path,
        layer_name='block5_conv3',
        top_n=3,
        localized_cls1=0,
        localized_cls2=0,
        only_prediction=False,
        only_cam=False,
        visualize=True,
        alpha=0.5,
        save=True,
        save_dir='.',
        save_name='test.png',
        print_func=lambda x: print('[INFO]:', end=' '),
        verbose=1,
):
    '''
    Makes prediction for an input image and then
    computes its saliency for the input image using Attribution-analytical approaches: Guided-GradCAM

    ## argments:
    - layer_name: layer whose gradient will used when Grad-CAM and Guided-Backprobagation computed
    - top_n: number of predictions with high probability to be shown
    - localized_cls1: class number to localize for output1 (0 for most probable class, and 1 for the second most probable class, and so on)
    - localized_cls2: class number to localize for output2
    - only_prediction: specifies model will produce only prediction or not
    - only_cam: specifies results will be explained only with Grad-CAM or even more with Guided-Backpropagation
    - visualize: specifies a result figure will be showed or not
    - alpha: transperency in Grad-CAM visualizeation
    '''
    # load and preprocess the input image
    preprocessed_input = load_image(
        image_path, target_size=(H, W), preprocessing=True)

    # make prediction
    if verbose:
        print_func()
        print('Making prediction ...')
    y1_pred, y2_pred = model.predict(preprocessed_input)
    sorted_indices1 = np.argsort(y1_pred[0])[::-1]
    sorted_indices2 = np.argsort(y2_pred[0])[::-1]

    # show prediction
    top_indices1 = sorted_indices1[:2]
    ps1 = list(map(lambda x: x * 100, y1_pred[0][top_indices1]))
    print_func()
    print('Prediction 1')
    for i, p in zip(top_indices1, ps1):
        print(' - {:<7s}: {:.3f} %'.format(LABEL2CLASS1[i][1].capitalize(), p))
    top_indices2 = sorted_indices2[:top_n]
    ps2 = list(map(lambda x: x * 100, y2_pred[0][top_indices2]))
    print_func()
    print('Prediction 2 for top {} cities'.format(top_n))
    for i, p in zip(top_indices2, ps2):
        tmp = LABEL2CLASS2[i][1]
        tmp_city_name = tmp.capitalize() if tmp != 'nyc' else 'NYC'
        print(' - {:<7s}: {:.3f} %'.format(tmp_city_name, p))

    # classes to be localized
    cls1 = sorted_indices1[localized_cls1]
    cls2 = sorted_indices2[localized_cls2]
    cls1_name = LABEL2CLASS1[cls1][1].capitalize()
    cls2_name = LABEL2CLASS2[cls2][1].capitalize(
    ) if LABEL2CLASS2[cls2][1] != 'nyc' else 'NYC'
    cls1_prob = ps1[localized_cls1]
    cls2_prob = ps2[localized_cls2]

    if not only_prediction:
        # make explainations
        if verbose:
            print_func()
            print('Explaining for \'{}\' and \'{}\' ...'.format(
                cls1_name, cls2_name, image_path))

        if only_cam:

            original_image = load_image(
                image_path, target_size=None, preprocessing=False)
            output_shape = original_image.size

            # Grad-CAM
            gc1 = compute_grad_cam(
                model,
                preprocessed_input,
                0,
                cls1,
                layer_name,
                output_shape=output_shape)
            gc2 = compute_grad_cam(
                model,
                preprocessed_input,
                1,
                cls2,
                layer_name,
                output_shape=output_shape)

            # set figure

            hw_ratio = output_shape[0] / output_shape[1]
            plt.figure(figsize=(hw_ratio * 4 * 3, 5), dpi=200)
            plt.subplot(131)
            plt.title('original')
            plt.axis('off')
            plt.imshow(original_image)
            plt.subplot(132)
            plt.title('Grad-CAM for \'{}\': {:.2f} %'.format(cls1_name, cls1_prob))
            plt.axis('off')
            plt.imshow(original_image)
            plt.imshow(gc1, cmap='jet', alpha=alpha)
            plt.subplot(133)
            plt.title('Grad-CAM for \'{}\': {:.2f} % '.format(cls2_name, cls2_prob))
            plt.axis('off')
            plt.imshow(original_image)
            plt.imshow(gc2, cmap='jet', alpha=alpha)
            plt.tight_layout()

        else:
            output_shape = (W, H)
            original_image = load_image(
                image_path, target_size=(H, W), preprocessing=False)

            # Grad-CAM
            gc1 = compute_grad_cam(
                model,
                preprocessed_input,
                0,
                cls1,
                layer_name,
                output_shape=output_shape)
            gc2 = compute_grad_cam(
                model,
                preprocessed_input,
                1,
                cls2,
                layer_name,
                output_shape=output_shape)
            # Guided-Backpropagation
            gb = guided_backpropagation(guided_model, preprocessed_input,
                                        layer_name)
            # Guided-Grad-CAM
            ggc1 = gb * gc1[..., np.newaxis]
            ggc2 = gb * gc2[..., np.newaxis]

            # set figure

            plt.figure(figsize=(12, 8), dpi=200)
            plt.subplot(231)
            plt.title('original')
            plt.axis('off')
            plt.imshow(original_image)
            plt.subplot(232)
            plt.title('Grad-CAM for \'{}\''.format(cls1_name))
            plt.axis('off')
            plt.imshow(original_image)
            plt.imshow(gc1, cmap='jet', alpha=alpha)
            plt.subplot(233)
            plt.title('Guided-Grad-CAM for \'{}\''.format(cls1_name))
            plt.axis('off')
            plt.imshow(np.flip(deprocess_image(ggc1[0]), -1))
            plt.subplot(234)
            plt.title('Guided-Backpropagation')
            plt.axis('off')
            plt.imshow(np.flip(deprocess_image(gb[0]), -1))
            plt.subplot(235)
            plt.title('Grad-CAM for \'{}\''.format(cls2_name))
            plt.axis('off')
            plt.imshow(original_image)
            plt.imshow(gc2, cmap='jet', alpha=alpha)
            plt.subplot(236)
            plt.title('Guided-Grad-CAM for \'{}\''.format(cls2_name))
            plt.axis('off')
            plt.imshow(np.flip(deprocess_image(ggc2[0]), -1))
            plt.tight_layout()

        if save:
            save_path = os.path.join(save_dir, save_name + '.png')
            try:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                if verbose:
                    print_func()
                    print(
                        'The result figure was saved as {}'.format(save_path))

            except FileNotFoundError:
                print(
                    '[INFO]: Saving figure failed, check the target directory exists'
                )

        if visualize:
            plt.show()
        else:
            plt.close()
