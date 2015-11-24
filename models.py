# -*- coding: utf-8 -*-
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from keras import activations, initializations

import optimizer
import cPickle

import numpy as np


def save_tparams(tparams, path):
    with open(path,'wb') as f:
        for params in tparams:
            cPickle.dump(params.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_tparams(tparams, path):
    with open(path,'rb') as f:
          for i in range(len(tparams)):
              tparams[i].set_value(cPickle.load(f))
    return tparams

def get_minibatch_indices(n, batch_size, shuffle=False):

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return minibatches


def dropout_layer(state_before, use_noise, trng, p):
    ratio = 1. - p
    proj = T.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=ratio, n=1, dtype=state_before.dtype), # for training..
            state_before * ratio)
    return proj

    
class RNN_GRU:
    def __init__(self, n_vocab, y_vocab, dim_word, dim):
        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512
        
        
        ### image Embedding
        self.W_img_emb = initializations.uniform((4096, self.dim))     
        self.b_img_emb = initializations.zero((self.dim))

   
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        ### enc forward GRU ###
        self.W_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.W_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim))       
        ### prediction ###
        self.W_pred = initializations.uniform((self.dim, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_img_emb, self.b_img_emb,
                       self.W_emb,
                       self.W_gru, self.U_gru, self.b_gru,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd,
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, init_state, mask=None):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd
        
        def _step(m_, x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_ # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return h#, r, u, preact, preactx
        seqs = [mask, state_below_, state_belowx]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state], #T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')
        x_mask = T.matrix('x_mask', dtype='float32')
        y = T.matrix('y', dtype = 'int32')
        img = T.matrix('img', dtype = 'float32')
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        init_state = T.dot(img, self.W_img_emb) + self.b_img_emb
        emb = self.W_emb[x.flatten()]
        
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, init_state, mask=x_mask)
    
        
        # hidden 들의 평균
        proj = (proj * x_mask[:, :, None]).sum(axis=0)
        proj = proj / x_mask.sum(axis=0)[:, None]  # sample * dim
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        


        
        output = T.dot(proj, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)

        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
        
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, x_mask, img, y, cost, updates, prediction
        
        
 
        

    def train(self, train_x, train_mask_x, train_img, train_y,
              valid_x=None, valid_mask_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout=None,
              batch_size=16,
              epoch=100,
              save=None):

        n_train = train_x.shape[1]
        
        trng, use_noise, x, x_mask, img, y, cost, updates, prediction = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        train_model = theano.function(inputs=[x, x_mask, img, y],
                                      outputs=cost,
                                      updates=updates)
 
 
        valid_model = theano.function(inputs=[x, x_mask, img], outputs=prediction)   
        valid_batch_indices = get_minibatch_indices(valid_x.shape[1], batch_size)                                       

        best = 0.                                      
        
        #valid_batch_indices = self.get_minibatch_indices(valid_x.shape[1], batch_size)
        for i in xrange(epoch):
            
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
           
            for j, indices in enumerate(batch_indices):
                
                x = [ train_x[:,t] for t in indices]
                x = np.transpose(x)
                x_mask = [ train_mask_x[:,t] for t in indices]
                x_mask = np.transpose(x_mask)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t,:] for t in indices]
                img = np.array(img)
                
                minibatch_avg_cost = train_model(x, x_mask, img, y)
            print 'cost : ' , '\' in epoch \'', (i+1) ,'\' ]'
            
            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[:,t] for t in valid_indices]
                        val_x = np.transpose(val_x)
                        val_x_mask = [ valid_mask_x[:,t] for t in valid_indices]
                        val_x_mask = np.transpose(val_x_mask)
                        val_img = [ valid_img[t,:] for t in valid_indices]
                        val_img = np.array(val_img)
                        
                        valid_prediction += valid_model(val_x, val_x_mask, val_img).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best : 
                        best = float(correct) / len(valid_prediction)
                        print 'save param..',
                        save_tparams(self.params, 'model_simple.pkl')
                        print 'Done'
                    
                    
        print 'best : ', best
                    
                    
        
                    
    def prediction(self, test_x, test_mask_x, test_img,
             # valid_x=None, valid_mask_x=None, 
              #valid_y=None, valid_mask_y=None,
             # optimizer=None,
              lr=0.001,
              batch_size=16):
        
        load_tparams(self.params, '/home/seonhoon/Desktop/workspace/ImageQA/data/model_simple.pkl')
        test_shared_x = theano.shared(np.asarray(test_x, dtype='int32'), borrow=True)
        #test_shared_y = theano.shared(np.asarray(test_y, dtype='int32'), borrow=True)
        test_shared_mask_x = theano.shared(np.asarray(test_mask_x, dtype='float32'), borrow=True)
        test_shared_img = theano.shared(np.asarray(test_img, dtype='float32'), borrow=True)

        n_test = test_x.shape[1]
        n_test_batches = int(np.ceil(1.0 * n_test / batch_size))
        
        index = T.lscalar('index')    # index to a case
        final_index = T.lscalar('final_index')
        
        
        trng, use_noise, x, x_mask, img, y, _, _, prediction = self.build_model(lr)
       
        batch_start = index * batch_size
        batch_stop = T.minimum(final_index, (index + 1) * batch_size)     

        test_model = theano.function(inputs=[index, final_index],
                                      outputs=prediction,
                                      givens={
                                         x: test_shared_x[:,batch_start:batch_stop],
                                         x_mask: test_shared_mask_x[:,batch_start:batch_stop],
                                         img: test_shared_img[batch_start:batch_stop,:]})
        
        
        prediction = []
        for minibatch_idx in xrange(n_test_batches):
            print minibatch_idx, ' / ' , n_test_batches
            prediction += test_model(minibatch_idx, n_test).tolist()
        return prediction
        
        
        
        
    
class BIRNN_GRU:
    
                       
    def __init__(self, n_vocab, y_vocab, dim_word, dim):
        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512

        ### image Embedding
        self.W_img_emb = initializations.uniform((4096, self.dim))     
        self.b_img_emb = initializations.zero((self.dim))

   
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        ### enc forward GRU ###
        self.W_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.W_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim))       
        ### prediction ###
        self.W_pred = initializations.uniform((self.dim * 2, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_img_emb, self.b_img_emb,
                       self.W_emb,
                       self.W_gru, self.U_gru, self.b_gru,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd,
                       self.W_pred, self.b_pred]




    def gru_layer(self, state_below, init_state, mask=None):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd
        
        def _step(m_, x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_ # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return h#, r, u, preact, preactx
        seqs = [mask, state_below_, state_belowx]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state], #T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
        
        
        
                   
    def build_model(self, lr=0.001, dropout=None):
        def concatenate(tensor_list, axis=0):
            concat_size = sum(tt.shape[axis] for tt in tensor_list)
            output_shape = ()
            for k in range(axis):
                output_shape += (tensor_list[0].shape[k],)
            output_shape += (concat_size,)
            for k in range(axis + 1, tensor_list[0].ndim):
                output_shape += (tensor_list[0].shape[k],)
            out = T.zeros(output_shape)
            offset = 0
            for tt in tensor_list:
                indices = ()
                for k in range(axis):
                    indices += (slice(None),)
                indices += (slice(offset, offset + tt.shape[axis]),)
                for k in range(axis + 1, tensor_list[0].ndim):
                    indices += (slice(None),)
        
                out = T.set_subtensor(out[indices], tt)
                offset += tt.shape[axis]
        
            return out
            
            
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')
        x_mask = T.matrix('x_mask', dtype='float32')
        y = T.matrix('y', dtype = 'int32')
        img = T.matrix('img', dtype = 'float32')
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]


        init_state = T.dot(img, self.W_img_emb) + self.b_img_emb
        emb = self.W_emb[x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        
        xr = x[::-1]
        xr_mask = x_mask[::-1]
        
        embr = self.W_emb[xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.dim_word])       

        proj = self.gru_layer(emb, init_state, mask=x_mask)
        projr = self.gru_layer(embr, init_state, mask=xr_mask)
    
        proj = concatenate([proj, projr[::-1]], axis=proj.ndim-1)
       
        # hidden 들의 평균
        proj = (proj * x_mask[:, :, None]).sum(axis=0)
        proj = proj / x_mask.sum(axis=0)[:, None]  # sample * dim
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        

        if dropout is not None :
            proj = dropout_layer(proj, use_noise, trng, dropout)
        
        output = T.dot(proj, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
        
        
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, x_mask, img, y, cost, updates, prediction
        
        
 
        

    def train(self, train_x, train_mask_x, train_img, train_y,
              valid_x=None, valid_mask_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout=None,
              batch_size=16,
              epoch=100,
              save=None):

        n_train = train_x.shape[1]
        
        trng, use_noise, x, x_mask, img, y, cost, updates, prediction = self.build_model(lr, dropout)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        train_model = theano.function(inputs=[x, x_mask, img, y],
                                      outputs=cost,
                                      updates=updates)
        if valid is not None:
            valid_model = theano.function(inputs=[x, x_mask, img],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[1], batch_size)                                       

        best_p = 0.
        best_epoch = 0
        #valid_batch_indices = self.get_minibatch_indices(valid_x.shape[1], batch_size)
        for i in xrange(epoch):
            
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
           
            for j, indices in enumerate(batch_indices):
                
                x = [ train_x[:,t] for t in indices]
                x = np.transpose(x)
                x_mask = [ train_mask_x[:,t] for t in indices]
                x_mask = np.transpose(x_mask)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t] for t in indices]
                img = np.array(img)
                
                minibatch_avg_cost = train_model(x, x_mask, img, y)
            print 'cost : ' , minibatch_avg_cost, ' [ epoch \'', (i+1) ,'\' ]'
            

            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[:,t] for t in valid_indices]
                        val_x = np.transpose(val_x)
                        val_x_mask = [ valid_mask_x[:,t] for t in valid_indices]
                        val_x_mask = np.transpose(val_x_mask)
                        val_img = [ valid_img[t] for t in valid_indices]
                        val_img = np.array(val_img)

                        valid_prediction += valid_model(val_x, val_x_mask, val_img).tolist()
                    correct = 0 

                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best_p : 
                        best_p = float(correct) / len(valid_prediction)
                        best_epoch = i
                        if i > 10:
                            print 'save param..',
                            save_tparams(self.params, 'model_simple.pkl')
                            print 'Done'
                                
        print 'best : ', best_p, ' in ', best_epoch
                    
                    
                    
                    
        
                    
    def prediction(self, test_x, test_mask_x, test_img, test_y,
             # valid_x=None, valid_mask_x=None, 
              #valid_y=None, valid_mask_y=None,
             # optimizer=None,
              lr=0.001,
              batch_size=16,
              epoch=100,
              save=None):
        
        load_tparams(self.params, 'model.pkl')
        test_shared_x = theano.shared(np.asarray(test_x, dtype='int32'), borrow=True)
        #test_shared_y = theano.shared(np.asarray(test_y, dtype='int32'), borrow=True)
        test_shared_mask_x = theano.shared(np.asarray(test_mask_x, dtype='float32'), borrow=True)
        test_shared_img = theano.shared(np.asarray(test_img, dtype='float32'), borrow=True)

        n_test = test_x.shape[1]
        n_test_batches = int(np.ceil(1.0 * n_test / batch_size))
        
        index = T.lscalar('index')    # index to a case
        final_index = T.lscalar('final_index')
        
        
        trng, use_noise, x, x_mask, img, y, _, _, prediction = self.build_model(lr)
       
        batch_start = index * batch_size
        batch_stop = T.minimum(final_index, (index + 1) * batch_size)     

        test_model = theano.function(inputs=[index, final_index],
                                      outputs=prediction,
                                      givens={
                                         x: test_shared_x[:,batch_start:batch_stop],
                                         x_mask: test_shared_mask_x[:,batch_start:batch_stop],
                                         img: test_shared_img[batch_start:batch_stop,:]})
        
        
        prediction = []
        for minibatch_idx in xrange(n_test_batches):
            print minibatch_idx, ' / ' , n_test_batches
            prediction += test_model(minibatch_idx, n_test).tolist()
        return prediction
        



class BIRNN_GRU2:
    def __init__(self, n_vocab, y_vocab, dim_word, dim):

        self.n_vocab = n_vocab  # 12047
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 1024
        self.dim_ctx = 4096  # 4096
        
        ### initial context
        self.W_img_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_img_init = initializations.zero((self.dim))


   
        
        ### Word Embedding ###        
        self.W_emb = initializations.uniform((self.n_vocab, self.dim_word))
        
        self.W_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        
        self.W_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim)) 

        ### prediction ###
        self.W_pred = initializations.uniform((self.dim * 2, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_img_init, self.b_img_init,
                       self.W_emb,
                       self.W_gru, self.U_gru, self.b_gru, 
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd, 
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, mask=None, init_state=None):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        if init_state is None :
            init_state = T.alloc(0., n_samples, dim)
         
        
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd
        
        def _step(m_, x_, xx_, h_):

            preact = T.dot(h_, self.U_gru)
            preact += x_ # samples * 1024

            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, self.U_gru_cdd)
            preactx *= r
            preactx += xx_ # samples * 512
         
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return [h]#, r, u, preact, preactx
            
        seqs = [mask, state_below_, state_belowx]
        outputs_info = [init_state]
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = outputs_info,
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
        def concatenate(tensor_list, axis=0):
            concat_size = sum(tt.shape[axis] for tt in tensor_list)
            output_shape = ()
            for k in range(axis):
                output_shape += (tensor_list[0].shape[k],)
            output_shape += (concat_size,)
            for k in range(axis + 1, tensor_list[0].ndim):
                output_shape += (tensor_list[0].shape[k],)
            out = T.zeros(output_shape)
            offset = 0
            for tt in tensor_list:
                indices = ()
                for k in range(axis):
                    indices += (slice(None),)
                indices += (slice(offset, offset + tt.shape[axis]),)
                for k in range(axis + 1, tensor_list[0].ndim):
                    indices += (slice(None),)
        
                out = T.set_subtensor(out[indices], tt)
                offset += tt.shape[axis]
        
            return out


    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')  # step * samples
        x_mask = T.matrix('x_mask', dtype='float32')  # step * samples
        y = T.matrix('y', dtype = 'int32')  # sample * emb
        img = T.matrix('img', dtype='float32')  # sample * 4096
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        xr = x[::-1]
        xr_mask = x_mask[::-1]
        
        emb = self.W_emb[x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])

        embr = self.W_emb[xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.dim_word])
        
 
        
        init_state = T.dot(img, self.W_img_init) + self.b_img_init
                
             
        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, mask=x_mask, init_state=init_state)
        proj_h = proj[0]

        projr = self.gru_layer(embr, mask=xr_mask, init_state=init_state)
        projr_h = projr[0]


        concat_proj_h = concatenate([proj_h, projr_h[::-1]], axis=proj_h.ndim-1)
        # step_ctx : step * samples * (dim*2)
        concat_proj_h = (concat_proj_h * x_mask[:,:,None]).sum(0) / x_mask.sum(0)[:,None]
        # step_ctx_mean : samples * (dim*2)


        if dropout is not None :
            concat_proj_h = dropout_layer(concat_proj_h, use_noise, trng, dropout)


        
        output = T.dot(concat_proj_h, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
    
    
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, x_mask, img, y, cost, updates, prediction
        
        
        
    def train(self, train_x, train_mask_x, train_img, train_y,
              valid_x=None, valid_mask_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout=None,
              batch_size=16,
              epoch=100,
              save=None):


        n_train = train_x.shape[1]

        trng, use_noise, x, x_mask, img, y, cost, updates, prediction = self.build_model(lr, dropout)
        # x : step * sample * dim
        # x_mask : step * sample
        # ctx : sample * annotation * dim_ctx
        # y : sample * emb
        

        train_model = theano.function(inputs=[x, x_mask, img, y],
                                      outputs=cost,
                                      updates=updates)
        
        if valid is not None:
            valid_model = theano.function(inputs=[x, x_mask, img],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[1], batch_size)
            

        best = 0.
        for i in xrange(epoch):
            use_noise.set_value(1.)
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
            

            
            for j, indices in enumerate(batch_indices):
                minibatch_avg_cost = 0 
                
                x = [ train_x[:,t] for t in indices]
                x = np.transpose(x)
                x_mask = [ train_mask_x[:,t] for t in indices]
                x_mask = np.transpose(x_mask)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t] for t in indices] 
                img = np.array(img)
     
     
                   
                minibatch_avg_cost = train_model(x, x_mask, img, y)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', j+1, '\' in epoch \'', (i+1) ,'\' ]'


            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[:,t] for t in valid_indices]
                        val_x = np.transpose(val_x)
                        val_x_mask = [ valid_mask_x[:,t] for t in valid_indices]
                        val_x_mask = np.transpose(val_x_mask)
                        val_img = [ valid_img[t] for t in valid_indices]
                        val_img = np.array(img)
                        
                        valid_prediction += valid_model(val_x, val_x_mask, val_img).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best : 
                        best = float(correct) / len(valid_prediction)
                        if i > 10:
                            if (i+1) % save == 0:
                                print 'save param..',
                                save_tparams(self.params, 'model_simple.pkl')
                                print 'Done'
                                








                    
                    
                  
                  
                    
 