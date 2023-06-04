import os
import scanpy
import scib
import scipy
from keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from pathlib import Path

import random

# Set seed for reproducibility
seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)  # sets the hash seed for Python
random.seed(seed_value)  # sets the seed for Python's built-in random module
np.random.seed(seed_value)  # sets the seed for Numpy (which Scipy also relies on)
tf.random.set_seed(seed_value)  # sets the seed for TensorFlow


class ABC(keras.Model):
    """ABC (Autoencoder-based Batch Correction) is a semi-supervised deep learning architecture for integrating
    single cell sequencing datasets. This method removes batch effects through a guided process of data compression
    using supervised cell type classifier branches for biological signal retention. It aligns the different batches
    using an adversarial training approach.

    Attributes:
        adata (anndata): anndata object containing the normalized and log1 transformed count data in adata.X and
            the cell type and batch label assignments in adata.obs[type_label] and adata.obs[batch_label].
        batch_label (String): the batch label key in adata.obs.
        type_label (String): the cell type label key in adata.obs.
        latent_dim (int): length of the latent dimension vector the encoder will encode the data into. Default: 64.

        types (pandas dataframe): a pandas dataframe holding the unique cell type label values.
        n_types (int): the number of unique cell type label values.
        batches (pandas dataframe): a pandas dataframe holding the unique batch label values.
        n_batches (int): the number of unique batch label values.
        n_genes (int): the number of genes (adata.n_vars)

        dropout_r (float): the dropout rate of the Encoder's dropout layer. Default: 0.1.
        enc_loss_tracker (keras.metrics.Mean): a loss tracker for the Encoder loss. (name="encoder_loss")
        dec_loss_tracker (keras.metrics.Mean): a loss tracker for the Decoder loss. (name="decoder_loss")
        advs_loss_tracker (keras.metrics.Mean): a loss tracker for the Adversarial training loss. (name="advs_loss")
        disc_loss_tracker (keras.metrics.Mean): a loss tracker for the Discriminator loss. (name="disc_loss")
        model_loss_tracker (keras.metrics.Mean): a loss tracker for the Full model loss. (name="model_loss")

        input_l (tf.keras.layers.Input): the input layer of the architecture.
        encoder_out (tf.keras.layers.Dropout): the Encoder's output layer.
        encoder (keras.Model): the Encoder model.
        decoder_out (tf.keras.layers.Dense): the Decoder's output layer.
        decoder (keras.Model): the Decoder model.
        classifier_layer (tf.keras.layers.Dense): the Output Classifier's output layer.
        classifier (keras.Model): the Output Classifier model.
        latent_classifier_l (tf.keras.layers.Dense): the Latent Classifier's output layer.
        latent_classifier (keras.Model): the Latent Classifier model.
        disc_out (tf.keras.layers.Dense): the Discriminator's output layer.
        discriminator (keras.Model): the Discriminator model.
        model (keras.Model): the Main Model (includes the layers from the Encoder, Decoder, Output Classifier and
            Latent classifier).

        recon_loss_w (float): the weight of the reconstruction loss in the main model weighted loss equation.
        class_branch_loss_w (float): the weight of the classification losses in the main model weighted loss equation.
        dec_optimizer (keras.optimizers): the Decoder optimizer. Default: Adam.
        disc_optimizer (keras.optimizers): the Discriminator optimizer. Default: Adam.
        model_optimizer (keras.optimizers): the Main Model optimizer. Default: Adam.
        disc_loss (keras.losses): the Discriminator loss function. Default: CategoricalCrossentropy.
        advs_loss (keras.losses): the loss function for the adversarial training. Default: BinaryCrossentropy.
        deco_loss (keras.losses): the Decoder loss function. Default: MeanSquaredError.
        class_loss (keras.losses): the Cell-Type classifiers loss function. Default: CategoricalCrossentropy.

        checkpoint_p (String): the path to the checkpoint file (file containing the trained ABC model's weights).


    """

    def __init__(self, adata, batch_label, type_label, latent_dim=64):

        """Initializes ABC with the given attributes and keras.metrics.Mean loss trackers. This method also subsets
        the gene set to the top 3000 highly variable genes (if number of genes exceeds 3000) and defines the
        different sub-models (Encoder, Decoder, Cell type classifier branches, Main model and the Discriminator).

        Parameters:
            adata (anndata): anndata object containing the normalized and log1 transformed count data in adata.X and
                the cell type and batch label assignments in adata.obs[type_label] and adata.obs[batch_label].
            batch_label (String): the batch label key in adata.obs.
            type_label (String): the cell type label key in adata.obs.
            latent_dim (int): length of the latent dimension vector the encoder will encode the data into. Default: 64.
        """

        super(ABC, self).__init__()

        # Preprocessing (batch aware 3000 highly variable genes selection and conversion to dense matrix if necessary)
        # if the number of genes exceeds 3000, work only on the top 3000 highly variable genes
        if adata.n_vars > 3000:
            adata = scib.preprocessing.hvg_batch(adata, batch_key=batch_label, target_genes=3000, adataOut=True)

        # if the given expression matrix is sparse, convert it to a dense matrix.
        if scipy.sparse.issparse(adata.X):
            print("The given adata.X matrix is sparse. Converting to dense.")
            adata.X = adata.X.todense()


        # Set model parameters and training trackers
        self.adata = adata        
        self.types = adata.obs[type_label].unique()
        self.n_types = len(self.types)
        self.batches = adata.obs[batch_label].unique()
        self.n_batches = len(self.batches)
        self.n_genes = adata.n_vars
        self.latent_dim = latent_dim
        self.batch_label = batch_label
        self.type_label = type_label
        self.dropout_r = 0.1
        self.enc_loss_tracker = keras.metrics.Mean(name="encoder_loss")
        self.dec_loss_tracker = keras.metrics.Mean(name="decoder_loss")
        self.advs_loss_tracker = keras.metrics.Mean(name="advs_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="classifier_loss")
        self.model_loss_tracker = keras.metrics.Mean(name="model_loss")
        self.model_loss_tracker.update_state([1])


        # sanity check
        if self.latent_dim > self.n_genes:
            print("Latent dimention larger than input (number of genes)."
                  "Setting latent dimention to half the input size.")
            self.latent_dim = int(self.n_genes/2)

        # Define the size of hidden layers
        hidden_size = int((self.latent_dim + self.n_genes) / 2)
        hidden_size = min(1500, hidden_size)


        # --- Encoder Definition ---
        self.input_l = tf.keras.layers.Input(shape=(self.n_genes,), name='input')
        hidden_l = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2())(self.input_l)
        latent_l = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2())(hidden_l)
        self.encoder_out = tf.keras.layers.Dropout(self.dropout_r, name='encoderOut')(latent_l)
        self.encoder = keras.Model(self.input_l, self.encoder_out, name="encoder")


        # --- Decoder Definition ---
        x = self.encoder_out
        hidden_l = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2())(x)

        self.decoder_out = tf.keras.layers.Dense(self.n_genes, name="decoderOut",
                                         kernel_regularizer=tf.keras.regularizers.l2())(hidden_l)
        self.decoder = keras.Model(self.input_l, self.decoder_out, name="decoder")


        # --- Output Cell-type Classifier Definition ---
        classifier_layer = tf.keras.layers.Dense(self.n_genes, activation=tf.nn.relu,
                                                 kernel_regularizer=tf.keras.regularizers.l2())(self.decoder_out)
        classifier_layer = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                                                 kernel_regularizer=tf.keras.regularizers.l2())(classifier_layer)

        self.classifier_layer = tf.keras.layers.Dense(self.n_types, activation=tf.nn.softmax, name="classifier",
                                         kernel_regularizer=tf.keras.regularizers.l2())(classifier_layer)
        self.classifier = keras.Model(self.input_l, self.classifier_layer, name="classifier")


        # --- Latent Cell-type Classifier Definition ---
        latent_classifier_l = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2())(self.encoder_out)
        self.latent_classifier_l = tf.keras.layers.Dense(self.n_types, activation=tf.nn.softmax,
                                        name="latent_classifier",
                                        kernel_regularizer=tf.keras.regularizers.l2())(latent_classifier_l)
        self.latent_classifier = keras.Model(self.input_l, self.latent_classifier_l, name="latent_classifier")


        # --- Discriminator Definition ---
        self.disc_in_genes = tf.keras.layers.Input(shape=(self.n_genes,))
        self.disc_in_types = tf.keras.layers.Input(shape=(self.n_types,))

        disc_input = tf.keras.layers.concatenate([self.disc_in_genes, self.disc_in_types])
        hidden_l = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2())(disc_input)
        hidden_l = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2())(hidden_l)

        self.disc_out = tf.keras.layers.Dense(self.n_batches, activation=tf.nn.softmax,
                                              kernel_regularizer=tf.keras.regularizers.l2())(hidden_l)
        self.discriminator = keras.Model(inputs=[self.disc_in_genes, self.disc_in_types],
                                         outputs=self.disc_out, name="discriminator")


        # --- Full (main) Model Definition ---
        self.model = keras.Model(inputs=self.input_l, outputs=[self.classifier_layer,
                                                               self.decoder_out,
                                                               self.latent_classifier_l])


    @property
    def metrics(self):
        return [self.enc_loss_tracker, self.dec_loss_tracker, self.disc_loss_tracker,
                self.class_loss_tracker,
                self.model_loss_tracker, self.advs_loss_tracker]


    # noinspection PyMethodOverriding
    def compile(self,
                dec_optimizer,
                disc_optimizer,
                model_optimizer,
                recon_loss_w=0.2,
                disc_loss=keras.losses.CategoricalCrossentropy(),
                advs_loss=keras.losses.BinaryCrossentropy(),
                deco_loss=keras.losses.MeanSquaredError(),
                class_loss=keras.losses.CategoricalCrossentropy()
                ):

        """Initializes ABC with the given attributes and compile the sub-models.

        Parameters:
            recon_loss_w (float): the weight of the reconstruction loss in the main model weighted loss equation.
            class_branch_loss_w (float): the weight of the classification losses in the main model weighted loss equation.
            dec_optimizer (keras.optimizers): the optimizer for training the Decoder. Default: Adam.
            disc_optimizer (keras.optimizers): the optimizer for training the Discriminator. Default: Adam.
            model_optimizer (keras.optimizers): the optimizer for training the Main Model. Default: Adam.
            disc_loss (keras.losses): the loss function for training the Discriminator. Default: CategoricalCrossentropy.
            advs_loss (keras.losses): the loss function for the adversarial training. Default: BinaryCrossentropy.
            deco_loss (keras.losses): the loss function for training the Decoder. Default: MeanSquaredError.
            class_loss (keras.losses): the loss function for training the Cell-Type classifiers. Default: CategoricalCrossentropy.
        """

        super(ABC, self).compile()
        self.recon_loss_w = recon_loss_w                # the weight of the reconstruction error in the main model loss
        self.class_branch_loss_w = (1-recon_loss_w)/2  # the weight of the classification error in the main model loss
        self.dec_optimizer = dec_optimizer
        self.disc_optimizer = disc_optimizer
        self.model_optimizer = model_optimizer
        self.disc_loss = disc_loss
        self.advs_loss = advs_loss
        self.deco_loss = deco_loss
        self.class_loss = class_loss

        self.model.compile(loss={'classifier': self.class_loss, 'decoderOut': self.deco_loss,
                                 'latent_classifier': self.class_loss},
                           loss_weights={'classifier': self.class_branch_loss_w,
                                         'decoderOut': self.recon_loss_w,
                                         'latent_classifier': self.class_branch_loss_w},
                           optimizer=self.model_optimizer)
        self.discriminator.compile(loss=self.disc_loss, optimizer=self.disc_optimizer)
        self.decoder.compile(loss=self.deco_loss, optimizer=self.dec_optimizer)


    # return two tensors: adata.x dataset and a tensor with two columns: the cell's type and the cell's batch,
    # where they are encoded as int by taking the index into the list of types and of batches.
    # the given adata needs to be a subset of the adata that initialized the model. If the given adata is None
    # the adata is the one that initialized the model.
    def get_dataset(self, shuffle=False, weights=False, subsample=None, adata=None):

        """creates and returns the dataset and labels for the training process from either the given adata anndata
        object, or from the self.adata anndata object which was initialized during the creation of the model.

        Parameters:
            shuffle (bool): If true the dataset is shuffled. Default: False.
            weights (bool): If true, the sample weights are calculated using sklearn compute_sample_weight() and the
                resulting ndarray of shape (n_samples,) which holds the sample weights is also returned.
            subsample (float): If not None, the dataset is subsampled to this fraction. Default: None.
            adata (anndata): if not None, the dataset is created from it, else, the dataset is created from self.adata
                which was initialized during the creation of the model. Default: None.

        Returns:
            data: A tensor holding the count data (adata.X converted to a tensor).
            labels: A tensor holding the cell type and batch labels as int values (the String labels are converted to
                int by taking their index into the list of unique labels).
            sample_weights: if weights is True, ndarray of sample weights is returned.

        """

        if adata is None:
            adata = self.adata

        if subsample:
            adata = scanpy.pp.subsample(adata, fraction=subsample, random_state=1, copy=True)

        # get expression matrix from adata
        x = adata.X

        # convert string type labels to int labels by taking their indexes into the list of all types
        t_string_labels = adata.obs[self.type_label]
        all_types = self.types.astype(str).tolist()
        t_int_labels = [all_types.index(s) for s in t_string_labels]
        b_string_labels = adata.obs[self.batch_label]
        all_batches = self.batches.astype(str).tolist()
        b_int_labels = [all_batches.index(s) for s in b_string_labels]

        # combine both labels lists to a dataframe
        labels = pd.DataFrame(list(zip(t_int_labels, b_int_labels)))

        # convert to tensors and float 32
        data = tf.convert_to_tensor(x)
        labels = tf.convert_to_tensor(labels)

        if shuffle:
            idxs = tf.range(tf.shape(x)[0])
            ridxs = tf.random.shuffle(idxs)
            data = tf.gather(data, ridxs)
            labels = tf.gather(labels, ridxs)

        if weights:
            t_labels = tf.gather(labels, 0, axis=1)
            sample_weights = class_weight.compute_sample_weight('balanced', t_labels)
            return data, labels, sample_weights

        else:
            return data, labels


    def corrected_data(self, checkpoint_path=None, adata=None):

        """Corrects the batch effects by using the trained Decoder on the original data and returns the batch corrected
        integrated data as a new anndata object.

        Parameters:
            checkpoint_path (String): The path to the saved models weights. If not None, the model weights are loaded
                from the checkpoint path and the dataset is decoded using the resulting model, else, the current model
                weights are used. Default: None.
            adata (anndata): If not None, the dataset being corrected is taken from it, else, the self.adata object
                dataset (which was initialized during the creation of the model) is corrected. Default: None.

        Returns:
            new_adata: A new anndata object with the batch corrected dataset.
        """


        if adata is None:
            adata = self.adata

        if checkpoint_path:
            self.load_weights(checkpoint_path)

        new_adata = adata.copy()
        data, _ = self.get_dataset(shuffle=False, adata=adata)
        new_adata.X = self.decoder(data)

        return new_adata


    def train_step(self, dataset):

        """The training step function for the entire architecture. The training is divided into three steps, in each one
        a selected set of weights are updated: First the Discriminator is trained to predict the given cell’s batch
        label, then the Autoencoder is trained, updating the weights of both the Encoder and Decoder sub-models (the
        Autoencoder is trained in an adversarial manner, opposing the Discriminator) and finally the main model’s
        weights are updated, using the weighted loss function combining the loss functions of all three main model
        outputs.

        Parameters:
            dataset (tuple): A tuple holding a tensor of the original dataset training batch, the cell type and batch
                labels tensor, and the ndarray of sample weights if they are used (these are the items returned from
                the get_dataset function, wrapped in a tuple by the fit function.

        Returns:
            metrics: A dictionary of the model's loss values (holds a key and the current value for each loss tracker).

        """


        # Unpack the data.
        if len(dataset) == 3:
            orig_data, labels, sample_weight = dataset
        else:
            sample_weight = None
            orig_data, labels = dataset

        type_labels = tf.gather(labels, 0, axis=1)
        batch_labels = tf.gather(labels, 1, axis=1)
        batch_size = tf.shape(orig_data)[0]

        # convert the type labels to 1-hot-encoded matrix
        hot_types = tf.one_hot(type_labels, depth=self.n_types)

        # convert the type labels to 1-hot-encoded matrix
        hot_batches = tf.one_hot(batch_labels, depth=self.n_batches)


        # Train the Discriminator
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator.trainable_variables)
            predictions = self.discriminator([self.decoder(orig_data), hot_types])
            dis_loss = self.discriminator.compiled_loss(hot_batches, predictions)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        self.disc_loss_tracker.update_state(dis_loss)

        # Train the autoencoder adversarially
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.decoder.trainable_variables)

            predictions = self.discriminator([self.decoder(orig_data), hot_types])

            # for each instance take it's correct label's prediction and store these predictions in a new vector
            y_masked = hot_batches * predictions
            y_compact = tf.math.reduce_sum(y_masked, 1)

            # create a zero vector to use in binary cross entropy to eliminate log(p(y))
            zero_vec = tf.fill([batch_size], 0)

            # the next line calculates (as average for all instances): -log(1-p(y))
            dec_loss = self.advs_loss(zero_vec, y_compact)

        grads = tape.gradient(dec_loss, self.decoder.trainable_variables)
        self.dec_optimizer.apply_gradients(zip(grads, self.decoder.trainable_variables))
        self.advs_loss_tracker.update_state(dec_loss)

        # Train the main model (encoder+decoder+classifier parts)
        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch(self.model.trainable_variables)
            output = self.model(orig_data)
            m_loss = self.model.compiled_loss({'classifier': hot_types, 'decoderOut': orig_data,
                                               'latent_classifier': hot_types}, output,
                                        sample_weight=sample_weight)

        grads = tape1.gradient(m_loss, self.model.trainable_weights)
        self.model_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.model.compiled_metrics.update_state({'classifier': hot_types, 'decoderOut': orig_data}, output)


        # Monitor loss.
        self.model_loss_tracker.update_state(m_loss)
        metrics = {m.name: m.result() for m in self.model.metrics}
        metrics["model_loss"] = self.model_loss_tracker.result()
        metrics["disc_loss"] = self.disc_loss_tracker.result()
        metrics["advs_loss"] = self.advs_loss_tracker.result()

        return metrics


    # correcting batch effects in the dataset loaded to this model, saves the corrected data and returns it
    # as a new anndata object.
    def batch_correction(self, load_trained=False, checkpoint_filepath=None, data_name='dataset',
                         base_LR=None, subsample_training=None, recon_loss_w=0.8, epochs=50,
                         early_stopping=False):

        """The batch correction method used to integrate and correct the given self.adata object initialized during
        the creation of the ABC model. This method corrects the batch effects, saves the trained model's weights and
        returns the corrected data as a new anndata object.

        Parameters:
            load_trained (bool): If True, the batch correction is performed using the model weights saved in the
                checkpoint_filepath path instead of using the current model's weights.
            checkpoint_filepath (String): A path for saving the weights of the ABC model during and after training.
                If None, a folder named 'checkpoint' is created in the current directory and the weights are saved to
                a file in it under the data_name.
            data_name (String): A name to be given to the model weights file. Default: Dataset.
            base_LR (float): A learning rate to be set in the optimizers used to train the sub-models. If None, a
                non-uniform approach is used, where all sub-models except the Discriminator are trained using a learning
                rate of 0.0001, and the Discriminator is trained using a learning rate of 0.001. Default: None.
            subsample_training (float): If not None, the anndata used to initialize the ABC model is subsamples to a
                fraction equal to subsample_training and the dataset created from it for training is a subset of the
                original data, else, the full dataset is used. Default: None.
            recon_loss_w (float): the weight of the reconstruction loss in the main model weighted loss equation.
                Default: 0.8.
            epochs (int): The number of epochs to train the ABC model. Default: 50.
            early_stopping (bool): If True, a Keras EarlyStopping callback is used, monitoring the output classifier's
                loss for early stopping (if the loss is not improving for 5 epochs, the training is stopped).

        Returns:
            corrected: A new anndata object with the batch corrected dataset.

        """


        # get current location to be used as path to save integrated data
        base_path = Path(__file__).resolve().parent
        balance_class_weights = True
        delta = 0.002
        train_batch_size = 64


        # adjust training parameters to a small dataset
        if self.adata.n_obs < 10000:
            print("Small dataset. Decreasing training batch size to 32.")
            train_batch_size = 32
            delta = 0.000


        # define checkpoint path for saving model weights during training
        if not checkpoint_filepath:
            checkpoint_filepath = os.path.join(base_path, "checkpoint", data_name)


        # create necessary folders (if they don't exist)
        Path(checkpoint_filepath).mkdir(parents=True, exist_ok=True)
        self.checkpoint_p = checkpoint_filepath

        # prepare the training dataset
        if balance_class_weights:
            dataset, labels, sample_weights = self.get_dataset(shuffle=True, weights=True, subsample=subsample_training)
        else:
            dataset, labels, sample_weights = self.get_dataset(shuffle=True, weights=False, subsample=subsample_training)


        # define optimizers and learning rates
        if base_LR:
            LR = base_LR
        else:
            LR = 0.0001

        dec_optimizer = keras.optimizers.Adam(learning_rate=LR)
        model_optimizer = keras.optimizers.Adam(learning_rate=LR)
        if base_LR:
            disc_optimizer = keras.optimizers.Adam(learning_rate=LR)
        else:
            disc_optimizer = keras.optimizers.Adam(learning_rate=LR * 10)

        self.compile(dec_optimizer, disc_optimizer, model_optimizer, recon_loss_w=recon_loss_w)


        # Train the model and save its weights
        if not load_trained:
            callbacks = []

            if early_stopping:

                # define callbacks
                callbacks.append(EarlyStopping(monitor='classifier_loss', mode='min', patience=5, min_delta=delta))
                save_best_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='classifier_loss',
                    mode='min',
                    verbose=1,
                    save_best_only=True)
                callbacks.append(save_best_callback)

            else:
                bathces_per_epoch = self.adata.n_obs/train_batch_size
                save_best_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='classifier_loss',
                    mode='min',
                    verbose=0,
                    save_freq=int(bathces_per_epoch * 20),
                    save_best_only=False)
                callbacks.append(save_best_callback)

            # train the model
            if balance_class_weights:
                self.fit(dataset, labels, batch_size=train_batch_size, sample_weight=sample_weights,
                                         epochs=epochs,
                                         callbacks=callbacks)
            else:
                self.fit(dataset, labels, batch_size=train_batch_size, epochs=epochs,
                                         callbacks=callbacks)


            # save trained model's weights (if early_stopping is True, the weights were already saved)
            if not early_stopping:
                self.save_weights(checkpoint_filepath, overwrite=True)


        # using an existing trained model
        else:
            self.load_weights(checkpoint_filepath)


        # correct the data (batch removal)
        corrected = self.corrected_data(checkpoint_path=checkpoint_filepath)

        # return the corrected data (as a new anndata object)
        print("Integration complete.")
        return corrected
