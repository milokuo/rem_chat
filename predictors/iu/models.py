import tensorflow as tf
import numpy as np
import tf_slim as slim
import os

RESIZE_AOI = 256
RESIZE_FINAL = 227

tf.compat.v1.disable_eager_execution()


class ImageCoder(object):
    """Reference from rude-carnie"""
    def __init__(self):
        # Create a single Session to run all image coding calls.
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self._sess = tf.compat.v1.Session(config=config)
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.compat.v1.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self.crop = tf.image.resize(self._decode_jpeg, (RESIZE_AOI, RESIZE_AOI))

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self.crop, feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def levi_hassner_bn(nlabels, images, pkeep, is_training):
    """Reference from rude-carnie"""
    
    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
    weight_decay = 0.0005
    weights_regularizer = tf.keras.regularizers.l2(0.5 * (weight_decay))

    with tf.compat.v1.variable_scope("LeviHassnerBN", "LeviHassnerBN", [images]) as scope:
        with slim.arg_scope(
                [slim.convolution2d, slim.fully_connected],
                weights_regularizer=weights_regularizer,
                biases_initializer=tf.compat.v1.constant_initializer(1.),
                weights_initializer=tf.compat.v1.random_normal_initializer(stddev=0.005),
                trainable=True):
            with slim.arg_scope(
                    [slim.convolution2d],
                    weights_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                conv1 = slim.convolution2d(images, 96, [7,7], [4, 4], padding='VALID', biases_initializer=tf.compat.v1.constant_initializer(0.), scope='conv1')
                pool1 = slim.max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
                conv2 = slim.convolution2d(pool1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2') 
                pool2 = slim.max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
                conv3 = slim.convolution2d(pool2, 384, [3, 3], [1, 1], padding='SAME', biases_initializer=tf.compat.v1.constant_initializer(0.), scope='conv3')
                pool3 = slim.max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
                # can use tf.contrib.layer.flatten
                flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
                full1 = slim.fully_connected(flat, 512, scope='full1')
                drop1 = tf.nn.dropout(full1, (1 - (pkeep)), name='drop1')
                full2 = slim.fully_connected(drop1, 512, scope='full2')
                drop2 = tf.nn.dropout(full2, (1 - (pkeep)), name='drop2')

    with tf.compat.v1.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.random.normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)

    return output

def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):
    """Reference from rude-carnie"""

    if requested_step is not None:

        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)

def find_files(img_id):
    if os.path.exists(img_id): 
        return img_id
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        candidate = img_id + suffix
        if os.path.exists(candidate):
            return candidate
    return None

def make_multi_crop_batch(img_id, coder):
    """Reference and modified from rude-carnie project"""

    # Read the image file.
    with tf.compat.v1.gfile.FastGFile(img_id, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if '.png' in img_id:
        print('Converting PNG to JPEG for %s' % img_id)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data)

    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(tf.image.per_image_standardization(crop))
    crops.append(tf.image.per_image_standardization(tf.image.flip_left_right(crop)))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(tf.image.per_image_standardization(cropped))
        flipped = tf.image.per_image_standardization(tf.image.flip_left_right(cropped))
        crops.append(tf.image.per_image_standardization(flipped))

    image_batch = tf.stack(crops)
    return image_batch

def classify_one_multi_crop(sess, label_list, softmax_output, images, image_file, coder=ImageCoder()):
    """Reference and modified from rude-carnie proejct"""
    try:
        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)
        
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
    
        for i in range(1, batch_sz):
            output = output + batch_results[i]
        
        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
    
        nlabels = len(label_list)
        if nlabels > 2:
            tmp = output[best]
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

            return best, tmp, second_best, output[second_best]

        return best, output[best], not best, 1.0
    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)

if __name__ == '__main__':
    pass