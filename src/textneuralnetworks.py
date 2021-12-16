# Dec 14, 2021
import copy
import numpy as np
import random
import torch
import time

import torch.nn.functional as F
import torch_optimizer as optim


class EarlyStopping(object):
    '''
    MIT License, Copyright (c) 2018 Stefano Nardo https://gist.github.com/stefanonardo
    es = EarlyStopping(patience=5)

    for epoch in range(n_epochs):
        # train the model for one epoch, on training set
        train_one_epoch(model, data_loader)
        # evalution on dev set (i.e., holdout from training)
        metric = eval(model, data_loader_dev)
        if es.step(metric):
            break  # early stop criterion is met, we can stop now
    '''

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


class TextNeuralNetwork():

    class Network(torch.nn.Module):
        def __init__(self, text_args, audio_args, n_hiddens_list, h_activation_f, n_outputs):
            """
                text_args = {'embed_args': {'num_embeddings': len(vocab), 'embedding_dim': 100}, # {'embeddings': embeddings}
                             'padding_idx': wtoi['<pad>'],
                             'lstm_args': {'lstm_hidden_dim': 64, 'n_lstm_layers': 1},
                             'cnn_args': [{'n_units': 5, 'window': 1}, {'n_units': 5, 'window': 2},
                                        {'n_units': 5, 'window': 3}, {'n_units': 5, 'window': 5}],
                             'activation_f': 'relu'
                            }
                audio_args = {'n_inputs': (1, 1200, 80),
                              'conv_layers': [{'n_units': 4, 'shape': 3},
                                              {'n_units': 8, 'shape': [3, 3]}],
                              'activation_f': 'relu'
                             }
                n_hiddens_list = [0], [10, 10], etc.
                n_outputs: int
            """
            super().__init__()
            # text_model
            #---------------------------------------------------------------#
            self.text_model = None
            text_ni = 0
            if text_args is not None:
                assert isinstance(text_args, dict), 'text_args must be a dict.'
                assert 'embed_args' in text_args and isinstance(
                    text_args['embed_args'], dict), 'text_args[\'embed_args\'] must be a dict.'
                assert 'padding_idx' in text_args, 'padding_idx must be specified in text_args.'
                text_ni = self.init_text(text_args)

            # audio_model
            #---------------------------------------------------------------#
            self.audio_model = None
            audio_ni = 0
            if audio_args is not None:
                assert isinstance(
                    audio_args, dict), 'audio_args must be a dict.'
                assert 'n_inputs' in audio_args and len(
                    audio_args['n_inputs']) == 3, 'n_inputs must be specified in audio_args.'
                audio_ni = self.init_audio(audio_args)

            # hidden_model
            #---------------------------------------------------------------#
            if not isinstance(n_hiddens_list, list):
                raise Exception(
                    'Network: n_hiddens_list must be a list.')

            if len(n_hiddens_list) == 0 or n_hiddens_list[0] == 0:
                self.n_hidden_layers = 0
            else:
                self.n_hidden_layers = len(n_hiddens_list)

            ni = text_ni + audio_ni

            # output network varaibles
            self.n_hiddens_list = n_hiddens_list
            self.n_outputs = n_outputs

            self.hidden_model = torch.nn.ModuleList()
            self.hidden_model.add_module(
                f'dropout', torch.nn.Dropout(0.2))  # !!!

            activation = self.get_activation(h_activation_f)

            l = 0
            # add fully-connected layers
            if self.n_hidden_layers > 0:
                for n_units in n_hiddens_list:
                    self.hidden_model.add_module(
                        f'linear_{l}', torch.nn.Linear(ni, n_units))  # !!!
                    self.hidden_model.add_module(
                        f'activation_{l}', activation())  # !!!
                    ni = n_units
                    l += 1
            self.hidden_model.add_module(
                f'output_{l}', torch.nn.Linear(ni, n_outputs))

            # self.audio_model.apply(self._init_weights)

        def get_activation(self, activation_f):
            activations = [
                torch.nn.Tanh,
                torch.nn.Sigmoid,
                torch.nn.ReLU,
                torch.nn.ELU,
                torch.nn.PReLU,
                torch.nn.ReLU6,
                torch.nn.LeakyReLU,
                torch.nn.Mish,
            ]
            activnames = [str(o.__name__).lower() for o in activations]
            try:
                return activations[activnames.index(str(activation_f).lower())]
            except:
                raise NotImplementedError(
                    f'__init__: {activation_f=} is not yet implemented.')

        def init_text(self, d):
            self.text_model = torch.nn.ModuleList()
            self.padding_idx = d['padding_idx']
            try:
                activation_f = d['activation_f']
            except KeyError:
                activation_f = 'tanh'
            activation = self.get_activation(activation_f)

            if 'embeddings' in d['embed_args']:
                # embed_args = {'embeddings': embeddings}
                embedding = torch.nn.Embedding.from_pretrained(
                    torch.from_numpy(d['embed_args']['embeddings']).float(),
                    freeze=True, padding_idx=self.padding_idx)  # !!!
            else:
                # embed_args = {'num_embeddings': num_embeddings, 'embedding_dim', embedding_dim}
                assert d['embed_args'].keys() & {
                    'num_embeddings', 'embedding_dim'}, 'embed_args must contain `num_embeddings` \
                        and `embedding_dim`'
                embedding = torch.nn.Embedding(d['embed_args']['num_embeddings'], d['embed_args']['embedding_dim'],
                                               padding_idx=self.padding_idx)

            self.num_embeddings = embedding.num_embeddings
            self.embedding_dim = embedding.embedding_dim
            self.text_model.add_module(f'embedding', embedding)  # !!!

            if 'lstm_args' in d:
                # lstm_args = {'lstm_hidden_dim': 64, 'n_lstm_layers': 1}
                self.modeltype = 0  # int for fast check
                self.lstm_hidden_dim = d['lstm_args'].get(
                    'lstm_hidden_dim', 64)
                self.n_lstm_layers = d['lstm_args'].get(
                    'n_lstm_layers', 1)
                self.text_model.add_module(f'lstm', torch.nn.LSTM(
                    input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim,
                    num_layers=self.n_lstm_layers, batch_first=True))  # !!!
                ni = self.lstm_hidden_dim
            elif 'cnn_args' in d:
                # cnn_args = [{'n_units': n_units, 'window': window}, ...]
                self.modeltype = 1
                convs = torch.nn.ModuleList()
                self.cnn_args = d['cnn_args']
                l = 0
                nc = 1
                for conv_layer in self.cnn_args:
                    convs.add_module(f'conv_{l}', torch.nn.Conv2d(nc, conv_layer['n_units'], (
                        conv_layer['window'], self.embedding_dim), stride=1,
                        padding='same', padding_mode='zeros'))
                    convs.add_module(f'activation_{l}', activation())
                    convs.add_module(f'maxpool_{l}', torch.nn.MaxPool2d(
                        (2, 1), stride=(2, 1)))
                    l += 1
                    nc = conv_layer['n_units']
                ni = nc * self.embedding_dim
                self.text_model.append(convs)  # !!!
                self.text_model.add_module(
                    'flatten', torch.nn.Flatten())  # !!!
            else:
                # do not specify `lstm_args` or `cnn_args`
                self.modeltype = 2
                ni = self.embedding_dim

            return ni

        def init_audio(self, d):
            self.audio_model = torch.nn.ModuleList()
            self.max_mel_len = d['n_inputs'][1]
            try:
                activation_f = d['activation_f']
            except KeyError:
                activation_f = 'tanh'
            activation = self.get_activation(activation_f)

            ni = np.asarray(d['n_inputs'])

            try:
                conv_layers = d['conv_layers']
            except KeyError:
                conv_layers = None
            l = 0
            if conv_layers is not None:
                for conv_layer in conv_layers:
                    n_channels = ni[0]  # C, H, W
                    self.audio_model.add_module(f'conv_{l}', torch.nn.Conv2d(
                        n_channels, conv_layer['n_units'], conv_layer['shape'],
                        stride=1, padding='same', padding_mode='zeros'))
                    self.audio_model.add_module(
                        f'activation_{l}', activation())
                    self.audio_model.add_module(
                        f'maxpool_{l}', torch.nn.MaxPool2d(2, stride=2))
                    # TODO: currently only to divide H, W dimensions by 2
                    # with 'same' padding
                    ni = np.concatenate([[conv_layer['n_units']], ni[1:] // 2])
                    l += 1
            self.audio_model.add_module('flatten', torch.nn.Flatten())  # okay
            ni = np.prod(ni)
            try:
                fc_layers = d['fc_layers']
            except KeyError:
                fc_layers = None
            # add fully-connected layers
            if fc_layers is not None:
                # self.audio_model.add_module(
                #     f'dropout', torch.nn.Dropout(0.1))  # !!!
                if len(fc_layers) != 0 and fc_layers[0] > 0:
                    for n_units in fc_layers:
                        self.audio_model.add_module(
                            f'linear_{l}', torch.nn.Linear(ni, n_units))  # !!!
                        self.audio_model.add_module(
                            f'activation_{l}', activation())  # !!!
                        ni = n_units
                        l += 1
            return ni

        def _init_weights(self, m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        def _init_lstm_hc(self, X):
            torch.manual_seed(1234)
            h = torch.zeros(
                (self.n_lstm_layers, X.size(0), self.lstm_hidden_dim))
            c = torch.zeros(
                (self.n_lstm_layers, X.size(0), self.lstm_hidden_dim))
            torch.nn.init.xavier_normal_(h)
            torch.nn.init.xavier_normal_(c)
            if next(self.text_model.lstm.parameters()).is_cuda:
                h = h.to('cuda')
                c = c.to('cuda')
            return h, c

        def forward_all_outputs(self, X, A, lens):
            XYs = []
            if X is not None:
                # embedding (bs, embedding_dim)
                XYs.append(self.text_model.embedding(X))
                if self.modeltype == 0:  # lstm
                    # ref https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
                    packed = torch.nn.utils.rnn.pack_padded_sequence(
                        XYs[-1], lens, batch_first=True, enforce_sorted=False)
                    lstm_hc = self._init_lstm_hc(X)  # (h,c)
                    out, (hs, cs) = self.text_model.lstm(packed, lstm_hc)
                    # lstm, (bs, embedding_dim)
                    XYs.append(hs[-1])
                elif self.modeltype == 1:  # cnn
                    XYs[-1] = torch.unsqueeze(XYs[-1], 1)
                    for layer in self.text_model[1]:
                        XYs.append(layer(XYs[-1]))
                    # mean over timesteps (BS, C, T, E) -> (BS, C, E)
                    XYs.append(torch.mean(XYs[-1], dim=2))
                    XYs.append(self.text_model.flatten(XYs[-1]))
                elif self.modeltype == 2:
                    # mean over timesteps (BS, T, E) -> (BS, E)
                    XYs.append(torch.mean(XYs[-1], dim=1))
                else:
                    raise NotImplementedError(
                        f'Forward pass not implemented for {self.modeltype=}')

            AYs = []
            if A is not None:
                for i, layer in enumerate(self.audio_model):
                    if i == 0:
                        AYs.append(layer(A))
                    else:
                        AYs.append(layer(AYs[-1]))

            Ys = XYs + AYs
            if len(XYs) > 0 and len(AYs) > 0:
                Ys.append(torch.cat((XYs[-1], AYs[-1]), dim=1))

            for layer in self.hidden_model:
                Ys.append(layer(Ys[-1]))
            return Ys

        def forward(self, X, A, lens):
            Ys = self.forward_all_outputs(X, A, lens)
            return Ys[-1]

    def __init__(self, text_args, audio_args, n_hiddens_list, n_outputs,
                 h_activation_f='tanh', use_gpu=True, seed=None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed

        if use_gpu and not torch.cuda.is_available():
            print('\nGPU is not available. Running on CPU.\n')
            use_gpu = False
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu else 'cpu')

        # Build nnet
        self.model = self.Network(
            text_args, audio_args, n_hiddens_list, h_activation_f, n_outputs)

        self.model.to(self.device)
        self.loss = None
        self.optimizer = None

        # Variables for standardization
        self.Amax = None
        self.Amin = None
        self.Tmeans = None
        self.Tstds = None

        # Bookkeeping
        self.train_error_trace = []
        self.val_error_trace = []
        self.n_epochs = None
        self.batch_size = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.model.n_outputs=},'
        str += f' {self.use_gpu=}, {self.seed=})'
        if self.training_time is not None:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds.\n   Final objective values...'
            str += f' train: {self.train_error_trace[-1]:.3f},'
            if len(self.val_error_trace):
                str += f'val: {self.val_error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str

    def summary(self):
        print(self.model)

    def _standardizeA(self, A):
        return (2 * (A - self.Amin) / (self.Amax - self.Amin)) - 1

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def _setup_standardize(self, A, T):
        if A is not None and self.Amax is None:
            self.Amax = max([a.max() for a in A if a.shape[0] != 0])
            self.Amin = min([a.min() for a in A if a.shape[0] != 0])

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1

    def _padtextbatch(self, X):
        maxlen = max(map(len, X))
        pads, lens = [], []
        for x in X:
            pads.append(x + [self.model.padding_idx]*(maxlen-len(x)))
            lens.append(len(x))
        return torch.LongTensor(pads).to(self.device), lens

    def _padimbatch(self, A, out):
        for i, a in enumerate(A):
            a = self._standardizeA(a)
            pad = self.model.max_mel_len-a.shape[0]
            if pad > 0:  # ((top, bottom), (left, right))
                out[i, 0] = np.pad(a, ((0, pad), (0, 0)), 'constant')
            elif pad < 0:
                out[i, 0] = a[:pad]
            else:
                out[i, 0] = a
        return torch.from_numpy(out[:len(A)]).float().to(self.device)

    def _train(self, training_data, validation_data):
        # training
        #---------------------------------------------------------------#
        Xtrain, Atrain, Ttrain = training_data
        if Xtrain is None:
            X, lens = None, None
        if Atrain is None:
            A = None
        running_loss = 0
        self.model.train()
        if Atrain is not None:
            Aout = np.zeros((self.batch_size, 1,
                            self.model.max_mel_len, Atrain[0].shape[1]))
        for i in range(0, len(Ttrain), self.batch_size):
            if Xtrain is not None:
                X, lens = self._padtextbatch(Xtrain[i:i+self.batch_size])
            if Atrain is not None and not isinstance(Atrain, torch.Tensor):
                A = self._padimbatch(Atrain[i:i+self.batch_size], Aout)
            T = torch.LongTensor(
                Ttrain[i:i+self.batch_size]).flatten().to(self.device)
            # compute prediction error
            Y = self.model(X, A, lens)
            error = self.loss(Y, T)

            # backpropagation
            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()

            # unaveraged sum of losses over all samples
            # https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
            running_loss += error.item() * len(T)
        # maintain loss over every epoch
        self.train_error_trace.append(running_loss / len(Ttrain))

        # validation
        #---------------------------------------------------------------#
        if validation_data is not None:
            Xval, Aval, Tval = validation_data
            if Xval is None:
                X, lens = None, None
            if Aval is None:
                A = None
            running_loss = 0
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(Tval), self.batch_size):
                    if Xval is not None:
                        X, lens = self._padtextbatch(Xval[i:i+self.batch_size])
                    if Aval is not None and not isinstance(Aval, torch.Tensor):
                        A = self._padimbatch(Aval[i:i+self.batch_size], Aout)
                    T = torch.LongTensor(
                        Tval[i:i+self.batch_size]).flatten().to(self.device)
                    Y = self.model(X, A, lens)
                    error = self.loss(Y, T)
                    running_loss += error.item() * len(T)

                self.val_error_trace.append(running_loss / len(Tval))

    def train(self, Xtrain, Atrain, Ttrain, n_epochs, batch_size, learning_rate,
              opt='adam', weight_decay=0, early_stopping=False,
              validation_data=None, shuffle=False, verbose=True):

        if Xtrain is not None:
            assert len(Xtrain) == len(
                Ttrain), 'Xtrain and Ttrain must have equal dim 0.'
        if Atrain is not None:
            assert len(Atrain) == len(
                Ttrain), 'Atrain and Ttrain must have equal dim 0.'
        if Xtrain is not None and Atrain is not None:
            assert len(Xtrain) == len(Atrain) == len(
                Ttrain), 'Xtrain, Atrain, and Ttrain must have equal dim 0.'

        self._setup_standardize(Atrain, Ttrain)  # only occurs once

        if validation_data is not None:
            assert len(
                validation_data) == 3, 'validation_data: must be (Xval, Aval, Tval) or (Xval, None, Tval).'
            Xval, Aval, Tval = validation_data[0], validation_data[1], validation_data[2]

        if not isinstance(self.loss, (torch.nn.NLLLoss, torch.nn.CrossEntropyLoss)):
            Ttrain = self._standardizeT(Ttrain)
            if validation_data is not None:
                Tval = self._standardizeT(Tval)

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if self.loss is None:
            self.loss = torch.nn.MSELoss()

        if self.optimizer is None:
            optimizers = [
                torch.optim.SGD,
                torch.optim.Adam,
                optim.A2GradExp,
                optim.A2GradInc,
                optim.A2GradUni,
                optim.AccSGD,
                optim.AdaBelief,
                optim.AdaBound,
                optim.AdaMod,
                optim.Adafactor,
                optim.AdamP,
                optim.AggMo,
                optim.Apollo,
                optim.DiffGrad,
                optim.Lamb,
                optim.NovoGrad,
                optim.PID,
                optim.QHAdam,
                optim.QHM,
                optim.RAdam,
                optim.Ranger,
                optim.RangerQH,
                optim.RangerVA,
                optim.SGDP,
                optim.SGDW,
                optim.SWATS,
                optim.Yogi,
            ]
            names = [str(o.__name__).lower() for o in optimizers]
            try:
                self.optimizer = optimizers[names.index(str(opt).lower())](
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            except:
                raise NotImplementedError(
                    f'train: {opt=} is not yet implemented.')

        print_every = n_epochs // 10 if n_epochs > 9 else 1
        if early_stopping:
            es = EarlyStopping(patience=10)

        # training loop
        #---------------------------------------------------------------#
        print('Training Started!')
        start_time = time.time()
        for epoch in range(n_epochs):
            if shuffle:  # shuffle after every epoch
                if self.seed is not None:
                    random.seed(self.seed + epoch)
                if Atrain is None:
                    c = list(zip(Xtrain, range(len(Ttrain))))
                    random.shuffle(c)
                    Xtrain, inds = zip(*c)
                    Ttrain = Ttrain[list(inds)]
                elif Xtrain is None:
                    c = list(zip(Atrain, range(len(Ttrain))))
                    random.shuffle(c)
                    Atrain, inds = zip(*c)
                    Ttrain = Ttrain[list(inds)]
                else:
                    c = list(zip(Xtrain, Atrain, range(len(Ttrain))))
                    random.shuffle(c)
                    Xtrain, Atrain, inds = zip(*c)
                    Ttrain = Ttrain[list(inds)]
                del c

            # forward, grad, backprop
            self._train((Xtrain, Atrain, Ttrain), (Xval, Aval, Tval)
                        if validation_data is not None else None)
            if early_stopping and validation_data is not None and es.step(self.val_error_trace[-1]):
                self.n_epochs = epoch + 1
                break  # early stop criterion is met, we can stop now
            if verbose and (epoch + 1) % print_every == 0:
                st = f'Epoch {epoch + 1} error - train: {self.train_error_trace[-1]:.5f},'
                if validation_data is not None:
                    st += f' val: {self.val_error_trace[-1]:.5f}'
                print(st)
        self.training_time = time.time() - start_time

        # remove data from gpu, needed?
        torch.cuda.empty_cache()

        # convert loss to likelihood
        # TODO: append values to continue with training
        if isinstance(self.loss, (torch.nn.NLLLoss, torch.nn.CrossEntropyLoss)):
            self.train_error_trace = np.exp(
                -np.asarray(self.train_error_trace))
            if validation_data is not None:
                self.val_error_trace = np.exp(
                    -np.asarray(self.val_error_trace))

        return self

    def use(self, X, A, all_output=False):
        """TODO: implemented but not tested..."""
        Ys = None
        torch.manual_seed(1234)  # needed for lstm hc init
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                if A is not None:
                    Aout = np.zeros((self.batch_size, 1,
                                     self.model.max_mel_len, A[0].shape[1]))
                    n_samples = len(A)
                else:
                    n_samples = len(X)
                for i in range(0, n_samples, self.batch_size):
                    Xt, lens = self._padtextbatch(
                        X[i:i+self.batch_size]) if X is not None else (None, None)
                    At = self._padimbatch(
                        A[i:i+self.batch_size], Aout) if A is not None else None
                    Y = self._unstandardizeT(self.model.forward(
                        Xt, At, lens)).detach().cpu().numpy()
                    Ys = Y if Ys is None else np.vstack((Ys, Y))
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        return Ys


class TextNeuralNetworkClassifier(TextNeuralNetwork):

    class Network(TextNeuralNetwork.Network):
        def __init__(self, *args, **kwargs):
            super(self.__class__, self).__init__(*args, **kwargs)
            # not needed if CrossEntropyLoss is used.
            # self.model.add_module(f'log_softmax', torch.nn.LogSoftmax(dim=1))

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # TODO: only supporting CrossEntropyLoss as use function now computes softmax
        # self.loss = torch.nn.NLLLoss()
        # CrossEntropyLoss computes LogSoftmax then NLLLoss
        self.loss = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)

    def use(self, X, A):
        """TODO: need to implement all_output for batches with variable length seq."""
        Ys = None
        torch.manual_seed(1234)  # needed for lstm hc init
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                if A is not None:
                    Aout = np.zeros((self.batch_size, 1,
                                     self.model.max_mel_len, A[0].shape[1]))
                    n_samples = len(A)
                else:
                    n_samples = len(X)
                for i in range(0, n_samples, self.batch_size):
                    Xt, lens = self._padtextbatch(
                        X[i:i+self.batch_size]) if X is not None else (None, None)
                    At = self._padimbatch(
                        A[i:i+self.batch_size], Aout) if A is not None else None
                    Y = F.softmax(self.model.forward(
                        Xt, At, lens), dim=1).detach().cpu().numpy()  # probabilities
                    Ys = Y if Ys is None else np.vstack((Ys, Y))
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        return Ys.argmax(1).reshape(-1, 1)


if __name__ == '__main__':
    print('Tests are not implemented')
    # import matplotlib.pyplot as plt
    # plt.switch_backend('tkagg')

    # def rmse(A, B): return np.sqrt(np.mean((A - B)**2))
    # def accuracy(A, B): return 100. * np.mean(A == B)
    # br = ''.join(['-']*8)

    # print(f'{br}Testing NeuralNetwork for regression{br}')
    #---------------------------------------------------------------#
    # X = np.arange(100).reshape((-1, 1))
    # T = np.sin(X * 0.04)

    # n_hiddens_list = [10, 10]

    # nnet = NeuralNetwork(X.shape[1], n_hiddens_list,
    #                      T.shape[1], activation_f='tanh')
    # nnet.summary()
    # nnet.train(X, T, n_epochs=1000, batch_size=32,
    #            learning_rate=0.01, opt='sgd')
    # Y = nnet.use(X)

    # print(f'RMSE: {rmse(T, Y):.3f}')
    # # plt.plot(nnet.train_error_trace)
    # # plt.show()

    # print(f'{br}Testing NeuralNetwork for CNN regression{br}')
    # #---------------------------------------------------------------#
    # # TODO: requires C, H, W dimensions
    # X = np.zeros((100, 1, 10, 10))
    # T = np.zeros((100, 1))
    # for i in range(100):
    #     col = i // 10
    #     X[i, :, 0:col + 1, 0] = 1
    #     T[i, 0] = col + 1

    # conv_layers = [{'n_units': 1, 'shape': [3, 3]},
    #                {'n_units': 1, 'shape': [3, 3]}]
    # n_hiddens_list = [10]

    # nnet = NeuralNetwork(X.shape[1:], n_hiddens_list,
    #                      T.shape[1], conv_layers, activation_f='tanh')
    # nnet.summary()
    # nnet.train(X, T, n_epochs=1000, batch_size=32,
    #            learning_rate=0.001, opt='adam')
    # Y = nnet.use(X)

    # print(f'RMSE: {rmse(T, Y):.3f}')
    # # plt.plot(nnet.train_error_trace)
    # # plt.show()

    # print(f'{br}Testing NeuralNetwork for CNN classification{br}')
    # #---------------------------------------------------------------#
    # X = np.zeros((100, 1, 10, 10))
    # T = np.zeros((100, 1))
    # for i in range(100):
    #     col = i // 10
    #     X[i, 0, :, 0:col + 1] = 1
    #     # TODO: class must be between [0, num_classes-1]
    #     T[i, 0] = 0 if col < 3 else 1 if col < 7 else 2

    # n_hiddens_list = [5]*2
    # conv_layers = [{'n_units': 3, 'shape': 3},
    #                {'n_units': 1, 'shape': [3, 3]}]

    # nnet = NeuralNetworkClassifier(X.shape[1:], n_hiddens_list, len(
    #     np.unique(T)), conv_layers, use_gpu=True, seed=None)
    # nnet.summary()
    # nnet.train(X, T, validation_data=None,
    #            n_epochs=50, batch_size=32, learning_rate=0.01, opt='adam',  # accsgd
    #            ridge_penalty=0, verbose=True)
    # Y = nnet.use(X)
    # print(f'Accuracy: {accuracy(Y, T)}:.3f')
    # # plt.plot(nnet.train_error_trace)
    # # plt.show()
