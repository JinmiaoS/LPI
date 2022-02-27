
import numpy as np

from layeruntil import AttentionWithContext, Addition

from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from keras.layers import Input, Dense,RepeatVector,TimeDistributed,Add
from keras.engine.training import Model

from keras.layers.recurrent import LSTM



def create_lstm_vae0(input_dim,X_train_tmp,X_test_tmp, timesteps,  intermediate_dim, latent_dim):

   
        batch_size=256 #40
       
        # timesteps, features
        input_x = Input(shape= (timesteps, input_dim)) 
        
        #intermediate dimension 
        h1 = LSTM(intermediate_dim, recurrent_activation='hard_sigmoid',recurrent_dropout=0.0,dropout=0.5, kernel_initializer='glorot_uniform', return_sequences=True)(input_x)
        h2 = LSTM(intermediate_dim//2, recurrent_activation='hard_sigmoid',recurrent_dropout=0.0,dropout=0.5, kernel_initializer='glorot_uniform', return_sequences=True)(h1)
        h3 = LSTM(intermediate_dim//4, recurrent_activation='hard_sigmoid',recurrent_dropout=0.0,dropout=0.5, kernel_initializer='glorot_uniform', return_sequences=True)(h2)
        ha =AttentionWithContext()(h3)
        haa =Addition()(ha)

        # Reconstruction decoder
        decoder0 = RepeatVector(timesteps)(haa)
        decoder1 = LSTM(intermediate_dim//4, recurrent_activation='hard_sigmoid',recurrent_dropout=0.0,dropout=0.5, kernel_initializer='glorot_uniform',  return_sequences=True)(decoder0)
        decoder1 = Add()([decoder1, h3])
        decoder1=AttentionWithContext()(decoder1)
        decoder2 = LSTM(intermediate_dim//2, recurrent_activation='hard_sigmoid',recurrent_dropout=0.0,dropout=0.5, kernel_initializer='glorot_uniform',  return_sequences=True)(decoder1)
        decoder2 = Add()([decoder2, h2])
        decoder2=AttentionWithContext()(decoder2)
        decoder3 = LSTM(intermediate_dim, recurrent_activation='hard_sigmoid',recurrent_dropout=0.0,dropout=0.5, kernel_initializer='glorot_uniform',  return_sequences=True)(decoder2)
        decoder3 = Add()([decoder3, h1])
        decodera=AttentionWithContext()(decoder3)
        decoder = TimeDistributed(Dense(input_dim))(decodera)
        
       
        m = Model(input_x, decoder)
        # m.add_loss(vae_loss2(input_x, decoder1, z_log_sigma, z_mean)) #<===========
        m.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
        m.summary()
        early_stopping = EarlyStopping(
            monitor='loss',
            min_delta=0.0001,
            patience=5,
            verbose=0,
            mode='auto'
        )

        history = m.fit(X_train_tmp, X_train_tmp, epochs=100, callbacks = [early_stopping], batch_size=batch_size)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc='lower right')
        plt.figure()

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


        X_train_tmp = m.predict(X_train_tmp)
        X_test_tmp = m.predict(X_test_tmp)
        return X_train_tmp,X_test_tmp

def autoencoder_two_subnetwork_LSTM_fine_tuning_1(X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = None, batch_size =128, nb_epoch = 60):
    print ('autoencode learning')
    last_dim = 64

    intermediate_dim=256

    timesteps=1
    
    XX_train1 = np.reshape(X_train1, (X_train1.shape[0], 1, X_train1.shape[1]))
    XX_train2 = np.reshape(X_train2, (X_train2.shape[0], 1, X_train2.shape[1]))
    XX_test1 = np.reshape(X_test1, (X_test1.shape[0], 1, X_test1.shape[1]))
    XX_test2 = np.reshape(X_test2, (X_test2.shape[0], 1, X_test2.shape[1]))
    
    conbine_train= np.concatenate((X_train1, X_train2), axis = 1) 
    conbine_test= np.concatenate((X_test1, X_test2), axis = 1) 
    XX_conbine_train = np.reshape(conbine_train, (conbine_train.shape[0], 1, conbine_train.shape[1]))
    XX_conbine_test = np.reshape(conbine_test, (conbine_test.shape[0], 1, conbine_test.shape[1]))
         

    X_train1_tmp_bef, X_test1_tmp_bef = create_lstm_vae0(X_train1.shape[1], XX_train1, XX_test1, timesteps, intermediate_dim,  last_dim)
    X_train2_tmp_bef, X_test2_tmp_bef = create_lstm_vae0(X_train2.shape[1], XX_train2, XX_test2, timesteps, intermediate_dim,  last_dim)
    
    conbine_train_bef, conbine_test_bef = create_lstm_vae0(conbine_train.shape[1], XX_conbine_train, XX_conbine_test, timesteps, intermediate_dim*2,  last_dim*2)
    
    print(X_train1_tmp_bef.shape, X_train2_tmp_bef.shape)
    
    X_train1_tmp_bef = np.squeeze(X_train1_tmp_bef , 1)
    X_train2_tmp_bef = np.squeeze(X_train2_tmp_bef , 1)
    X_test1_tmp_bef = np.squeeze(X_test1_tmp_bef , 1)
    X_test2_tmp_bef = np.squeeze(X_test2_tmp_bef , 1)
    prefilter_train_bef = np.concatenate((X_train1_tmp_bef, X_train2_tmp_bef), axis =1)
    prefilter_test_bef = np.concatenate((X_test1_tmp_bef, X_test2_tmp_bef), axis = 1)
    
    conbine_train_bef = np.squeeze(conbine_train_bef , 1)
    conbine_test_bef = np.squeeze(conbine_test_bef , 1)
    
    return conbine_train_bef, conbine_test_bef, prefilter_train_bef, prefilter_test_bef, X_train1_tmp_bef, X_test1_tmp_bef, conbine_train,conbine_test
