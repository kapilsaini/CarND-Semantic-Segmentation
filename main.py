import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from glob import glob
from tqdm import tqdm


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph = tf.get_default_graph()
    
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer_3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer_4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer_7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer_3_out_tensor, vgg_layer_4_out_tensor, vgg_layer_7_out_tensor

#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully 	convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    new_layer_7 = tf.layers.conv2d(inputs=vgg_layer7_out, filters=num_classes, kernel_size=(1,1),
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='new_layer_7',
                                activation=tf.nn.relu)
    
    new_layer_7_up = tf.layers.conv2d_transpose(inputs=new_layer_7, filters=num_classes, kernel_size=(3, 3),
                                padding='same', strides=(2, 2),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='new_layer_7_up',
                                activation=tf.nn.relu)

    
    new_layer_4 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=(1, 1),
                                    padding='same', strides=(1, 1),
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                    name="new_layer_4",
                                    activation=tf.nn.relu)
                                            
    new_layer_4_7 = tf.add(new_layer_7_up, new_layer_4, name="new_layer_4_7")
        
    new_layer_4_7_up = tf.layers.conv2d_transpose(new_layer_4_7, filters=num_classes, kernel_size=(3, 3),
                                     strides=(2, 2), padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                     activation=tf.nn.relu,
                                     name="new_layer_4_7_up")
       
    new_layer_3= tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=(1, 1), 
                                  strides=(1, 1), padding='same', 
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), 
                                  activation=tf.nn.relu, 
                                  name="new_layer_3")
            
    new_layer_final  = tf.add(new_layer_3, new_layer_4_7_up)
   
   
    new_layer_final_up = tf.layers.conv2d_transpose(new_layer_final, filters=num_classes, kernel_size=(16, 16),
                                strides=(8, 8), padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                activation=tf.nn.relu,
                                name="new_layer_final_up")
        
    return new_layer_final_up
    
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label), name="cross_entropy")
                                        
    reshaped_logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    reshaped_correct_label = tf.reshape(correct_label, (-1, num_classes))
    
    is_correct_prediction = tf.equal(tf.argmax(reshaped_logits, 1), tf.argmax(reshaped_correct_label, 1))

    accuracy_op = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name="accuracy_op")
                            
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(cross_entropy_loss, name="train_op")
    
    return reshaped_logits, train_op, cross_entropy_loss
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """   
    
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
      for X_batch, y_batch in get_batches_fn(batch_size):
        loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
          input_image: X_batch,
          correct_label: y_batch,
          keep_prob: 0.8
        })
      print('Training loss ->', loss)
    
    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    
    tests.test_for_kitti_dataset(data_dir)
    
    # Hyper parameters
    EPOCHS = 20
    BATCH_SIZE = 6
    LEARN_RATE = 0.0001
   

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:    

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network        

        # TODO: Build NN using load_vgg, layers, and optimize function
        vgg_path = os.path.join(data_dir, 'vgg')
        vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer_3_out_tensor, \
            vgg_layer_4_out_tensor, vgg_layer_7_out_tensor = load_vgg(sess, vgg_path)
        
        output_tensor = layers(vgg_layer_3_out_tensor, vgg_layer_4_out_tensor, vgg_layer_7_out_tensor, num_classes)
        
        correct_label = tf.placeholder(tf.int8, (None,) + image_shape + (num_classes,), name="correct_label")
        print('correct_label_shape', correct_label.shape)
        
        learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

        # TODO: Train NN using the train_nn function
        
        logits, train_op, cross_entropy_loss = optimize(output_tensor, correct_label, LEARN_RATE, num_classes)

        train_nn(sess, epochs=EPOCHS, batch_size=BATCH_SIZE, get_batches_fn=get_batches_fn, train_op=train_op, 
                 cross_entropy_loss=cross_entropy_loss, input_image=vgg_input_tensor, correct_label=correct_label,
                 keep_prob=vgg_keep_prob_tensor, learning_rate=LEARN_RATE)
                  

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                                      vgg_keep_prob_tensor, vgg_input_tensor)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()