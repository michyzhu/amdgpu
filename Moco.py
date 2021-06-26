import numpy as np
import keras
import tensorflow
from tensorflow.keras.optimizers import Adam                                                                                                                         
from keras.layers import Input, Conv3D, Dense, Flatten, Concatenate, Activation, Cropping3D, AveragePooling3D, GlobalAveragePooling3D, Dropout, MaxPooling3D
from tensorflow.keras.models import Model                                                                                                                                        
from keras.layers.normalization import BatchNormalization                                                                                                                       
# import keras.backend as K                                                                                                                                                   
import sys, pickle
import tensorflow as tf     

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
def moco_loss(tau):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    def moco_loss_actual(q, k, queue, labels):
        #inputs should be l2 normalized

        #k (B X C)
        #q (B X C)
        #queue (K X C)
        
        (B, C) = k.shape
        (K, _) = queue.shape

        l_pos = tf.reduce_sum(input_tensor=k * q, axis=-1, keepdims=True)
        # l_pos (B, 1)

        l_neg = tf.linalg.matmul(q, queue, transpose_b=True)
        # l_neg (B, K)

        l_all = tf.concat([l_pos, l_neg], axis=1) / tau
        # l_all (B, (K + 1))

        return tf.reduce_mean(input_tensor=cce(labels, l_all))

    return moco_loss_actual

    
#knowing i keeps us from having to do the costly operation of rotating the array
def enqueue_dequeue(queue, elem, i):
    queue_size = np.shape(queue)[0] 
    queue[i % queue_size, :, :] = elem
    return queue


#currently moco is implemented to ensure diversity
#in each batch through ensuring that exactly one
#element from each mini batch in the queue is in
#the current batch's dictionary
class Moco():
    #Requires encoder and decoder have same network structure
    def __init__(self, out_q, out_k, inp_q, inp_k, B, K, C, tau):
        #we train q and not k
        self.B = B # batch size
        self.K = K # num negative examples
        self.C = C # feature size

        self.f_k = Model(inputs=inp_k, outputs=out_k)
        self.f_q = Model(inputs=inp_q, outputs=out_q)
        self.f_k.set_weights(self.f_q.get_weights())

        self.loss_fun = moco_loss(tau)

    #requires augmenting function aug : a' -> np array
    def train(self, data, epochs, aug, m=0.999, learning_rate=1e-4, log_interval=None, folder="."):
        optimizer = tensorflow.keras.optimizers.SGD(learning_rate=learning_rate)

        B = self.B
        K = self.K
        C = self.C
        f_k = self.f_k
        f_q = self.f_q
        loss_fun = self.loss_fun

        num_batches = np.shape(data)[0] // B
        
        losses = []

        labels = np.zeros((B, K + 1))
        #Pos iff augmented image
        #by convention appears in first index
        labels[:, 0] = 1.
        for j in range(epochs):
            print("=== Started Epoch: ", str(j + 1), " ===")
            #create and fill queue
            perm = np.random.permutation(np.shape(data)[0])
            data = data[perm]
            queue = f_k.predict(aug(data[0 : K]))
            queue = queue.reshape((K // B, B, C))

            loss = 0
            for i in tqdm(range(K // B, num_batches)):
                #get current mini batch x functions as loader
                x = data[(B * i) : (B * (i + 1))]
                x_k = aug(x)
                x_q = aug(x)

                queue_copy = np.copy(queue).reshape((K, C))

                #no gradients for k
                # k = f_k.predict(x_k)
                k = f_k(x_k, training=True)

                with tf.GradientTape() as tape:
                    q = f_q(x_q, training=True)
                    loss_value = loss_fun(q, k, queue_copy, labels)

                grads = tape.gradient(loss_value, f_q.trainable_weights)
                optimizer.apply_gradients(zip(grads, f_q.trainable_weights))

                if log_interval != None:
                    if i % log_interval == 0:
                        print(float(loss_value))
                loss += float(loss_value)

                #momentum update
                weights_fk = self.f_k.get_weights()
                weights_fq = self.f_q.get_weights()
                weights_result = []
                for i in range(len(weights_fk)):
                    cur_weights = m * np.array(weights_fk[i]) + (1. - m) * np.array(weights_fq[i])
                    weights_result.append(cur_weights)
                self.f_k.set_weights(weights_result)

                #queue/dictionary update
                enqueue_dequeue(queue, k, i)
            print("Epoch Loss {:.2f}".format(loss/(num_batches - K // B)))
            losses.append(loss/(num_batches - K // B))
            plt.clf()
            plt.plot(list(range(len(losses))), losses)
            plt.title(f'moco loss: B = {B}, k = {K}, C = {C}')
            plt.xlabel("epochs")
            # newFolder = f'B={B}_{subtomosPerClass}per{numClasses}classes_snr-{snr}-inception_B={B}-k={k}-C={C}-epochs={epochs}-m={m}-tau={tau}'

            plt.savefig(folder + '/' + f'Moco_loss_B={B}_k={K}_C={C}.png')
            
































































