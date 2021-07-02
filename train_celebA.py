# train.py

import time
import torch
import torch.optim as optim
import plugins
import losses
import torch.nn as nn




class Trainer:
    def __init__(self, args, model, criterion, evaluation, lam, k, encoder):

        self.args = args
        self.encoder = encoder
        self.r = args.r
        self.total_classes = args.total_classes
        self.nclasses_A = args.nclasses_a
        self.nclasses_T = args.nclasses_t
        self.k = k

        self.adverserial_type = args.adverserial_type

        self.model = model
        self.criterion = criterion

        self.evaluation = evaluation


        self.save_results = args.save_results

        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.device = args.device
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size_train

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        self.lr_e = args.learning_rate_e
        self.optim_method_e = args.optim_method_e
        self.optim_options_e = args.optim_options_e
        self.scheduler_method_e = args.scheduler_method_e
        self.scheduler_options_e = args.scheduler_options_e

        self.lr_ea = args.learning_rate_ea
        self.optim_method_ea = args.optim_method_ea
        self.optim_options_ea = args.optim_options_ea
        self.scheduler_method_ea = args.scheduler_method_ea
        self.scheduler_options_ea = args.scheduler_options_ea

        self.lr_et = args.learning_rate_et
        self.optim_method_et = args.optim_method_et
        self.optim_options_et = args.optim_options_et
        self.scheduler_method_et = args.scheduler_method_et
        self.scheduler_options_et = args.scheduler_options_et

        self.lr_a = args.learning_rate_a
        self.optim_method_a = args.optim_method_a
        self.optim_options_a = args.optim_options_a
        self.scheduler_method_a = args.scheduler_method_a
        self.scheduler_options_a = args.scheduler_options_a

        self.lr_t = args.learning_rate_t
        self.optim_method_t = args.optim_method_t
        self.optim_options_t = args.optim_options_t
        self.scheduler_method_t = args.scheduler_method_t
        self.scheduler_options_t = args.scheduler_options_t

        if self.encoder == True:
            self.optimizer = {} # diifferent optim_options should be added

            self.optimizer['Encoder'] = getattr(optim, self.optim_method_e)(
                filter(lambda p: p.requires_grad, self.model['Encoder'].parameters()),
                lr=self.lr_e, **self.optim_options_e)

            self.optimizer['E-Adversary'] = getattr(optim, self.optim_method_ea)(
                filter(lambda p: p.requires_grad, self.model['E-Adversary'].parameters()),
                lr=self.lr_e, **self.optim_options_ea)

            self.optimizer['E-Target'] = getattr(optim, self.optim_method_et)(
                filter(lambda p: p.requires_grad, self.model['E-Target'].parameters()),
                lr=self.lr_e, **self.optim_options_et)

            self.scheduler = {}
            if self.scheduler_method_e is not None: # should have different scheduler for each model
                # import pdb; pdb.set_trace()
                self.scheduler['Encoder'] = getattr(optim.lr_scheduler, self.scheduler_method_e)(
                    self.optimizer['Encoder'], **self.scheduler_options_e
                )
                self.scheduler['E-Adversary'] = getattr(optim.lr_scheduler, self.scheduler_method_ea)(
                    self.optimizer['E-Adversary'], **self.scheduler_options_ea
                )
                self.scheduler['E-Target'] = getattr(optim.lr_scheduler, self.scheduler_method_et)(
                    self.optimizer['E-Target'], **self.scheduler_options_et
                )

        else:

            self.optimizer = {}

            self.optimizer['Adversary'] = getattr(optim, self.optim_method_a)(
                filter(lambda p: p.requires_grad, self.model['Adversary'].parameters()),
                lr=self.lr_a, **self.optim_options_a)

            self.optimizer['Target'] = getattr(optim, self.optim_method_t)(
                filter(lambda p: p.requires_grad, self.model['Target'].parameters()),
                lr=self.lr_t, **self.optim_options_t)

            self.scheduler = {}
            if self.scheduler_method_a is not None:
                self.scheduler['Adversary'] = getattr(optim.lr_scheduler, self.scheduler_method_a)(
                    self.optimizer['Adversary'], **self.scheduler_options_a
                )

            if self.scheduler_method_t is not None:
                self.scheduler['Target'] = getattr(optim.lr_scheduler, self.scheduler_method_t)(
                    self.optimizer['Target'], **self.scheduler_options_t
                )


        # for classification
        self.labels = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )

        self.sensitives = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.inputs = torch.zeros(
            self.batch_size,
            device=self.device
        )

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger_%d_%.4f.txt' %(k, lam),
            self.save_results
        )

        if self.encoder ==True:

            if args.adverserial_type == 'OptNet':
                self.params_loss = ['Loss', 'P_M*A', 'P_M*T']
                self.log_loss.register(self.params_loss)

                # monitor training
                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss': {'dtype': 'running_mean'},
                    'P_M*A': {'dtype': 'running_mean'},
                    'P_M*T': {'dtype': 'running_mean'}
                   }

                self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
                self.params_visualizer = {
                    'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss',
                             'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'P_M*A': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'P_M*A',
                               'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'P_M*T': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'P_M*T',
                               'layout': {'windows': ['train', 'test'], 'id': 0}},
                }


            if args.adverserial_type == 'SGDA' or args.adverserial_type ==  'ExtraSGDA':

                self.params_loss = ['Loss', 'Loss_EA', 'Loss_ET', 'Accuracy_EA', 'Accuracy_ET']
                self.log_loss.register(self.params_loss)

                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss': {'dtype': 'running_mean'},
                    'Loss_EA': {'dtype': 'running_mean'},
                    'Loss_ET': {'dtype': 'running_mean'},
                    'Accuracy_EA': {'dtype': 'running_mean'},
                    'Accuracy_ET': {'dtype': 'running_mean'},
                }

                self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
                self.params_visualizer = {
                    'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss',
                             'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Loss_EA': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_EA',
                                'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Loss_ET': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_ET',
                                'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Accuracy_EA': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_EA',
                                    'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Accuracy_ET': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_ET',
                                    'layout': {'windows': ['train', 'test'], 'id': 0}},
                }



        else:
            if args.adverserial_type == 'OptNet':
                self.params_loss = ['Loss_Adversary', 'Loss_Target', 'Accuracy_Adversary', 'Accuracy_Target']
                self.log_loss.register(self.params_loss)

                # monitor training
                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss_Adversary': {'dtype': 'running_mean'},
                    'Loss_Target': {'dtype': 'running_mean'},
                    'Accuracy_Adversary': {'dtype': 'running_mean'},
                    'Accuracy_Target': {'dtype': 'running_mean'},
                   }
                self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
                self.params_visualizer = {
                    'Loss_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Adversary',
                                    'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Loss_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Target',
                                    'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Accuracy_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Adversary',
                                        'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Accuracy_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Target',
                                        'layout': {'windows': ['train', 'test'], 'id': 0}},
                }
            if args.adverserial_type == 'SGDA' or args.adverserial_type =='ExtraSGDA':
                self.params_loss = ['Loss_Adversary', 'Accuracy_Adversary','Loss_Target', 'Accuracy_Target']
                self.log_loss.register(self.params_loss)

                # monitor training
                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss_Adversary': {'dtype': 'running_mean'},
                    'Accuracy_Adversary': {'dtype': 'running_mean'},
                    'Loss_Target': {'dtype': 'running_mean'},
                    'Accuracy_Target': {'dtype': 'running_mean'},
                }
                self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
                self.params_visualizer = {
                    'Loss_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Adversary',
                                       'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Accuracy_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Adversary',
                                           'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Loss_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Target',
                                    'layout': {'windows': ['train', 'test'], 'id': 0}},
                    'Accuracy_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Target',
                                        'layout': {'windows': ['train', 'test'], 'id': 0}},
                }


        self.monitor.register(self.params_monitor)
        self.visualizer.register(self.params_visualizer)

        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Train [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})'
            for item in self.params_loss:
                self.print_formatter += '|' + item + ' {:.3f}'
            self.print_formatter += '|lr:{:.2e}'
            self.print_formatter += '|lam:{:.2f}'
            self.print_formatter += '|adv_t:{:.10s}'
            self.print_formatter += '|enc_trn:{:b}'
            self.print_formatter += '|p:{:4d}'

        self.evalmodules = []
        self.losses = {}


    def model_train(self):

        if self.encoder==True:
            if self.adverserial_type == 'SGDA' or self.adverserial_type =='ExtraSGDA':
                self.model['Encoder'].train()
                self.model['E-Adversary'].train()
                self.model['E-Target'].train()
            elif self.adverserial_type == 'one-shot':
                self.model['E-Adversary'].train()
        else:
            self.model['Adversary'].train()
            self.model['Target'].train()



    def train(self, epoch, dataloader, lam, reg):
        dataloader = dataloader['train']
        self.monitor.reset()
        # switch to train mode
        self.model_train()

        if self.log_type == 'progressbar':
            # Progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<2}'.format('Train'), max=len(dataloader))
        end = time.time()


        for i, (inputs, labels, sensitives) in enumerate(dataloader): # Y, S


            ############################
            # Update network
            ############################

            batch_size = inputs.size(0)

            ############################################ if T <  A
            Y = torch.zeros(batch_size, self.total_classes).scatter_(1, labels.unsqueeze(1).long(), 1)
            S = torch.zeros(batch_size, self.total_classes).scatter_(1, sensitives.unsqueeze(1).long() +
                                                                     self.total_classes-self.nclasses_A, 1)
            self.inputs.resize_(inputs.size()).copy_(inputs)
            self.labels.resize_(labels.size()).copy_(labels)
            self.sensitives.resize_(sensitives.size()).copy_(sensitives)

            Y = Y.to(self.device)
            S = S.to(self.device)

            if self.encoder==True:

                self.outputs_E = self.model['Encoder'](self.inputs)


###################################################################################
                if self.adverserial_type == 'OptNet':

                    if self.args.loss_type_e == 'Projection':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.outputs_E, S, Y, lam, reg, self.device)
                    elif self.args.loss_type_e == 'Projection_poly':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.outputs_E, S, Y, lam, reg, self.device,
                                                                         self.args.c, self.args.d)
                    elif self.args.loss_type_e == 'Projection_gauss':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.outputs_E, S, Y, lam, reg, self.device,
                                                                         self.args.sigma)
                    elif self.args.loss_type_e == 'Projection_gauss_linear':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.outputs_E, S, Y, lam, reg, self.device,
                                                                         self.args.sigma)

                    self.optimizer['Encoder'].zero_grad()
                    loss.backward()
                    self.optimizer['Encoder'].step()

                    loss_A = loss_A.item()
                    loss_T = loss_T.item()
                    loss = loss.item()


                elif self.adverserial_type == 'SGDA':
                    outputs_A = self.model['E-Adversary'](self.outputs_E)
                    outputs_T = self.model['E-Target'](self.outputs_E)

                    loss_A = self.criterion['E-Discriminator'](outputs_A, S)
                    loss_T = self.criterion['E-Target'](outputs_T, Y)

                    loss = (1 - lam) * loss_T - lam * loss_A

                    self.optimizer['Encoder'].zero_grad()
                    self.optimizer['E-Target'].zero_grad()
                    self.optimizer['E-Adversary'].zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer['Encoder'].step()
                    self.optimizer['E-Target'].step()


                    self.optimizer['Encoder'].zero_grad()
                    self.optimizer['E-Target'].zero_grad()
                    self.optimizer['E-Adversary'].zero_grad()
                    loss_A.backward()
                    self.optimizer['E-Adversary'].step()

                    loss_A = loss_A.item()

                    loss_T = loss_T.item()

                    loss = loss.item()

                elif self.adverserial_type == 'ExtraSGDA':
                    outputs_A = self.model['E-Adversary'](self.outputs_E)
                    outputs_T = self.model['E-Target'](self.outputs_E)

                    loss_A = self.criterion['E-Discriminator'](outputs_A, S)
                    loss_T = self.criterion['E-Target'](outputs_T, Y)

                    loss = (1 - lam) * loss_T - lam * loss_A

                    # import pdb; pdb.set_trace()
                    self.optimizer['E-Adversary'].extrapolation()
                    self.optimizer['Encoder'].zero_grad()
                    self.optimizer['E-Target'].zero_grad()
                    self.optimizer['E-Adversary'].zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer['Encoder'].step()
                    self.optimizer['E-Target'].step()

                    self.optimizer['Encoder'].zero_grad()
                    self.optimizer['E-Target'].zero_grad()
                    self.optimizer['E-Adversary'].zero_grad()
                    loss_A.backward()
                    self.optimizer['E-Adversary'].step()

                    loss_A = loss_A.item()

                    loss_T = loss_T.item()

                    loss = loss.item()

##########################################################################
            else:
                self.model['Encoder'].eval()
                self.outputs_E = self.model['Encoder'](self.inputs).detach()

                if self.adverserial_type =='OptNet':

                    outputs_A = self.model['Adversary'](self.outputs_E)
                    outputs_T = self.model['Target'](self.outputs_E)

                    loss_A = self.criterion['Adversary'](outputs_A.squeeze(), S)
                    loss_T = self.criterion['Target'](outputs_T.squeeze(), Y)

                    self.optimizer['Adversary'].zero_grad()
                    self.optimizer['Target'].zero_grad()
                    loss_A.backward()
                    self.optimizer['Adversary'].step()

                    self.optimizer['Adversary'].zero_grad()
                    self.optimizer['Target'].zero_grad()
                    loss_T.backward()
                    self.optimizer['Target'].step()

                    loss_A = loss_A.item()
                    loss_T = loss_T.item()


                elif self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA':

                    outputs_A = self.model['Adversary'](self.outputs_E)
                    outputs_T = self.model['Target'](self.outputs_E)

                    loss_A = self.criterion['Adversary'](outputs_A.squeeze(), S)
                    loss_T = self.criterion['Target'](outputs_T.squeeze(), Y)

                    self.optimizer['Adversary'].zero_grad()
                    loss_A.backward()
                    self.optimizer['Adversary'].step()

                    self.optimizer['Target'].zero_grad()
                    loss_T.backward()
                    self.optimizer['Target'].step()

                    loss_A = loss_A.item()
                    loss_T = loss_T.item()

            ############################
            #   Evaluating
            ############################
            if self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA' or self.encoder==False:
                acc_A = self.evaluation['Adversary'](outputs_A, self.sensitives.long())
                acc_T = self.evaluation['Target'](outputs_T, self.labels.long())

                acc_A = acc_A.item()
                acc_T = acc_T.item()

            if self.encoder ==True:

                if self.adverserial_type == 'OptNet':
                    self.losses['Loss'] = loss
                    self.losses['P_M*A'] = loss_A
                    self.losses['P_M*T'] = loss_T

                if self.adverserial_type == 'SGDA' or self.adverserial_type =='ExtraSGDA':

                    self.losses['Loss'] = loss
                    self.losses['Loss_EA'] = loss_A
                    self.losses['Loss_ET'] = loss_T
                    self.losses['Accuracy_EA'] = acc_A
                    self.losses['Accuracy_ET'] = acc_T

                    nan = float('Nan')
                    if loss == nan or loss_A == nan or loss_T == nan:
                        import pdb; pdb.set_trace()

            else:

                if self.adverserial_type =='OptNet':
                    self.losses['Loss_Adversary'] = loss_A
                    self.losses['Loss_Target'] = loss_T
                    self.losses['Accuracy_Adversary'] = acc_A
                    self.losses['Accuracy_Target'] = acc_T


                if self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA':
                    self.losses['Loss_Adversary'] = loss_A
                    self.losses['Accuracy_Adversary'] = acc_A
                    self.losses['Loss_Target'] = loss_T
                    self.losses['Accuracy_Target'] = acc_T


            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                # import pdb; pdb.set_trace()
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]
                    +[self.optimizer['Encoder'].param_groups[-1]['lr']]))
            elif self.log_type == 'progressbar':
                if self.encoder == True:
                # update progress bar
                    batch_time = time.time() - end
                    processed_data_len += len(inputs)

                    bar.suffix = self.print_formatter.format(
                        *[processed_data_len, len(dataloader.sampler)]
                         + [self.losses[key] for key in self.params_monitor]
                         + [self.optimizer['Encoder'].param_groups[-1]['lr']]
                         + [lam]
                         + [self.adverserial_type]
                         + [self.encoder]
                         + [self.args.port]
                    )

                    bar.next()
                    end = time.time()
                    bar.finish()

                else:
                    processed_data_len += len(inputs)

                    bar.suffix = self.print_formatter.format(
                        *[processed_data_len, len(dataloader.sampler)]
                         + [self.losses[key] for key in self.params_monitor]
                         + [self.optimizer['Target'].param_groups[-1]['lr']]
                         + [lam]
                         + [self.adverserial_type]
                         + [self.encoder]
                         + [self.args.port]
                    )
                    bar.next()
                    end = time.time()
                    bar.finish()


        loss = self.monitor.getvalues()

        self.log_loss.update(loss)


        self.visualizer.update(loss)

        if self.encoder == True:
            if self.scheduler_method_e is not None:
                if self.scheduler_method_e == 'ReduceLROnPlateau':
                    self.scheduler['Encoder'].step(loss['Loss'])
                    self.scheduler['E-Target'].step(loss['Loss'])
                    self.scheduler['E-Adversary'].step(loss['Loss'])
                else:
                    self.scheduler['Encoder'].step()
                    self.scheduler['E-Target'].step()
                    self.scheduler['E-Adversary'].step()

        else:
            if self.scheduler_method_t is not None:
                if self.scheduler_method_t == 'ReduceLROnPlateau':
                    self.scheduler['Target'].step(loss['Loss'])
                else:
                    self.scheduler['Target'].step()

            if self.scheduler_method_a is not None:
                if self.scheduler_method_a == 'ReduceLROnPlateau':
                    self.scheduler['Adversary'].step(loss['Loss'])
                else:
                    self.scheduler['Adversary'].step()

        return loss
