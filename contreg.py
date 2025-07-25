import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, GlobalAveragePooling1D, LSTM, Dropout, Concatenate
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
import utils


def contrastive_regression(X_train, y_train, y_type_train, X_test, y_test, y_type_test):
    # Define triplet loss function
    def triplet_loss(anchor, positives, negatives, margin=4.0):
        # Reshape the inputs to combine the temporal and feature dimensions
        anchor_flat = tf.reshape(anchor, [anchor.shape[0], -1])
        positives_flat = tf.reshape(positives,
                                    [positives.shape[0], positives.shape[1], -1])
        negatives_flat = tf.reshape(negatives,
                                    [negatives.shape[0], negatives.shape[1], -1])

        # Normalize the vectors to unit length
        anchor_normalized = tf.nn.l2_normalize(anchor_flat, axis=-1)
        positives_normalized = tf.nn.l2_normalize(positives_flat, axis=-1)
        negatives_normalized = tf.nn.l2_normalize(negatives_flat, axis=-1)

        # Compute the cosine similarity
        pos_similarity = tf.reduce_sum(
            anchor_normalized[:, tf.newaxis, :] * positives_normalized, axis=-1)
        neg_similarity = tf.reduce_sum(
            anchor_normalized[:, tf.newaxis, :] * negatives_normalized, axis=-1)

        # Convert cosine similarity to cosine distance
        pos_distance = 1 - pos_similarity
        neg_distance = 1 - neg_similarity

        # Sum the distances to positives and negatives
        pos_distance_sum = tf.reduce_sum(pos_distance, axis=-1)
        neg_distance_sum = tf.reduce_sum(neg_distance, axis=-1)

        # Compute the triplet loss with cosine distance
        loss = tf.maximum(pos_distance_sum - neg_distance_sum + margin, 0.0)
        return tf.reduce_mean(loss)

    # Build GRU model with embedding and classification heads
    def build_contrastive_model(input_shape, num_lstm_layers, lstm_units, dense_units,
                                dropout_rate=0.3):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_lstm_layers):
            x = GRU(lstm_units, return_sequences=True)(x)
            x = Dropout(dropout_rate)(x)  # Add dropout after each GRU layer
        x = GlobalAveragePooling1D()(x)
        embeddings = Dense(dense_units, activation='relu')(x)

        # Create model
        model = Model(inputs, embeddings)
        return model

    # Build simple GRU model for normal learning
    def build_regression_model(input_shape, num_gru_layers, gru_units, dropout_rate=0.3):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_gru_layers):
            x = GRU(gru_units, return_sequences=True)(x)
            x = Dropout(dropout_rate)(x)  # Add dropout after each GRU layer
        x = GlobalAveragePooling1D()(x)
        mid = Dense(2, activation='relu')(x)
        classification_output = Dense(1)(mid)

        # Create model
        model = Model(inputs, classification_output)
        return model

    # Combine contrastive and normal learning models
    def combined_model(input_shape, num_layers, units, dense_units, dropout_rate=0.3):
        contrastive_model = build_contrastive_model(input_shape, num_layers, units,
                                                    dense_units, dropout_rate)
        regression_model = build_regression_model(input_shape, num_layers, units,
                                                  dropout_rate)

        inputs = Input(shape=input_shape)
        contrastive_embeddings = contrastive_model(inputs)
        regression_output = regression_model(inputs)

        combined_input = Concatenate()(
            [inputs[:, 0, :], contrastive_embeddings, regression_output])

        mid = Dense(12, activation='relu')(combined_input)

        mid = Dense(4, activation='relu')(mid)

        final_output = Dense(1, activation='sigmoid')(mid)

        model = Model(inputs, final_output)
        return model, contrastive_model, regression_model

    # Example usage
    input_shape = (60, 24)  # Update based on your actual data shape
    num_layers = 2
    num_units = 6
    dense_units = 4

    classification_model, contrastive_model, regression_model = combined_model(
        input_shape, num_layers, num_units, dense_units)
    contrastive_model.summary()
    regression_model.summary()
    classification_model.summary()

    # Function to generate triplets
    def generate_triplets(X, y, num_samples=4, batch_size=1024):
        anchors = []
        positives = []
        negatives = []

        num_batches = np.ceil(len(X) / batch_size).astype(int)

        for i in range(len(X)):
            anchor = X[i]
            g = i // batch_size

            # Calculate range for the current batch
            batch_start = batch_size * g
            batch_end = min(batch_size * (g + 1), len(X))

            positive_indices = np.where(
                (y == y[i]) & (np.arange(len(y)) >= batch_start) & (
                            np.arange(len(y)) < batch_end))[0]
            negative_indices = np.where(
                (y != y[i]) & (np.arange(len(y)) >= batch_start) & (
                            np.arange(len(y)) < batch_end))[0]

            positive_indices = positive_indices[positive_indices != i]

            selected_positives = list(positive_indices)
            selected_negatives = list(negative_indices)

            # Keep collecting positives and negatives until we have enough
            batch = g + 1
            while len(selected_positives) < num_samples or len(
                    selected_negatives) < num_samples:
                if batch >= num_batches:
                    batch = 0  # Loop back to the start of the dataset

                if len(selected_positives) < num_samples:
                    batch_start = batch_size * batch
                    batch_end = min(batch_size * (batch + 1), len(X))
                    new_positives = np.where(
                        (y == y[i]) & (np.arange(len(y)) >= batch_start) & (
                                    np.arange(len(y)) < batch_end))[0]
                    new_positives = new_positives[new_positives != i]
                    selected_positives.extend(new_positives)

                if len(selected_negatives) < num_samples:
                    batch_start = batch_size * batch
                    batch_end = min(batch_size * (batch + 1), len(X))
                    new_negatives = np.where(
                        (y != y[i]) & (np.arange(len(y)) >= batch_start) & (
                                    np.arange(len(y)) < batch_end))[0]
                    selected_negatives.extend(new_negatives)

                if batch == g:
                    # If we loop back to the original batch, break to avoid infinite loop
                    break
                batch += 1

            selected_positives = np.array(selected_positives)[:num_samples]
            selected_negatives = np.array(selected_negatives)[:num_samples]

            anchors.append(anchor)
            positives.append(X[selected_positives])
            negatives.append(X[selected_negatives])

        return np.array(anchors), np.array(positives), np.array(negatives)

    epochs = 10
    batch_size = 1024

    anchors, positives, negatives = generate_triplets(X_train, y_train, 4, batch_size)

    # Compile model with separate loss functions
    optimizer = tf.keras.optimizers.Adam()

    # Training loop with triplet and classification loss

    for epoch in range(epochs):
        epoch_loss_triplet = 0
        epoch_loss_regression = 0
        epoch_loss_classification = 0
        num_batches = 0

        print(f'Epoch {epoch + 1}/{epochs}')
        for i in tqdm(range(0, len(anchors), batch_size),
                      desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            a_batch = anchors[i:i + batch_size]
            p_batch = positives[i:i + batch_size]
            n_batch = negatives[i:i + batch_size]
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            y_type_batch = y_type_train[i:i + batch_size]

            # Reshape y_batch to match the shape of classification_output
            y_batch = y_batch.reshape(-1, 1)
            y_type_batch = y_type_batch.reshape(-1, 1)

            with tf.GradientTape() as tape:
                anchor_embeddings = contrastive_model(a_batch, training=True)

                positive_embeddings_list = []
                negative_embeddings_list = []

                for j in range(4):
                    positive_embedding = contrastive_model(p_batch[:, j, :, :],
                                                           training=True)
                    negative_embedding = contrastive_model(n_batch[:, j, :, :],
                                                           training=True)
                    positive_embeddings_list.append(positive_embedding)
                    negative_embeddings_list.append(negative_embedding)

                positive_embeddings = tf.stack(positive_embeddings_list, axis=1)
                negative_embeddings = tf.stack(negative_embeddings_list, axis=1)

                # Compute triplet loss
                loss_triplet = triplet_loss(anchor_embeddings, positive_embeddings,
                                            negative_embeddings)

                # Compute regression loss
                regression_output = regression_model(x_batch, training=True)
                loss_regression = tf.keras.losses.MeanSquaredError()(tf.convert_to_tensor(regression_output, dtype=tf.float32),
                                                                     regression_output)
                loss_regression = tf.reduce_mean(loss_regression) * 0.00000001

                # Compute final classification loss
                combined_output = classification_model(x_batch, training=True)
                loss_classification = tf.keras.losses.binary_crossentropy(y_batch,
                                                                          combined_output)
                loss_classification = tf.reduce_mean(loss_classification)

                # Total loss
                total_loss = loss_triplet + loss_regression + loss_classification

            gradients = tape.gradient(total_loss,
                                      classification_model.trainable_variables + regression_model.trainable_variables + contrastive_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          classification_model.trainable_variables + regression_model.trainable_variables + contrastive_model.trainable_variables))

            epoch_loss_triplet += loss_triplet.numpy()
            epoch_loss_regression += loss_regression.numpy()
            epoch_loss_classification += loss_classification.numpy()
            num_batches += 1

        avg_loss_triplet = epoch_loss_triplet / num_batches
        avg_loss_regression = epoch_loss_regression / num_batches
        avg_loss_classification = epoch_loss_classification / num_batches

        print(
            f'Epoch {epoch + 1} - Triplet Loss: {avg_loss_triplet:.4f}, Regression Loss: {avg_loss_regression:.4f}, Classification Loss: {avg_loss_classification:.4f}')

    print("Training completed!")

    best_threshold = 0.0
    best_tss = 0.0
    y_pred = classification_model.predict(X_test)
    # evaluate model
    for i in range(1, 1000):

        threshold = i / 1000  # Adjust the threshold as needed
        y_pred_binary = (y_pred > threshold).astype(int)
        confusion = confusion_matrix(y_test, y_pred_binary)
        tn, fp, fn, tp = confusion.ravel()
        metric = utils.Metric(cm=confusion)
        tss = metric.tss
        if tss > best_tss:
            best_tss = tss
            best_threshold = i / 1000

    print(str(X_train.shape) + ': The Classifier is Done! \n')

    threshold = best_threshold  # Adjust the threshold as needed
    y_pred_binary = (y_pred > threshold).astype(int)
    confusion = confusion_matrix(y_test, y_pred_binary)
    # tn, fp, fn, tp = confusion.ravel()
    # metric = utils.Metric(confusion)

    # tss = TSS(tp,tn,fp,fn)
    # hss1 = HSS1(tp,tn,fp,fn)
    # hss2 = HSS2(tp,tn,fp,fn)
    # gss = GSS(tp,tn,fp,fn)
    # recall = Recall(tp,tn,fp,fn)
    # precision = Precision(tp,tn,fp,fn)
    #
    # output_values = np.array([tp, fn, fp, tn, tss, hss1, hss2, gss, recall, precision])

    # joblib.dump(classifier, data_dir + "mlp_model.pkl")

    # loaded_mlp_model = joblib.load(data_dir + "mlp_model.pkl")

    return y_pred_binary