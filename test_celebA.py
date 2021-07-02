# test.py

import time
import torch
import plugins


class Tester:
    def __init__(self, args, model, criterion, evaluation, lam, k, encoder):
        self.args = args
        self.encoder = encoder
        self.total_classes = args.total_classes
        self.adverserial_type = args.adverserial_type
        self.nclasses_A = args.nclasses_a
        self.nclasses_T = args.nclasses_t

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
        self.batch_size = args.batch_size_test

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        # for classification
        self.labels = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.inputs = torch.zeros(
            self.batch_size,
            device=self.device
        )
        self.sensitives = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger_%1d_%.4f.txt' %(k, lam),
            self.save_results
        )
        if self.encoder ==True:
            if self.adverserial_type == 'OptNet':

                self.params_loss = ['Loss', 'P_M*A', 'P_M*T']
                self.log_loss.register(self.params_loss)

                # monitor testing
                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss': {'dtype': 'running_mean'},
                    'P_M*A': {'dtype': 'running_mean'},
                    'P_M*T': {'dtype': 'running_mean'}
                }

                self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
                self.params_visualizer = {
                    'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss',
                             'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'P_M*A': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'P_M*A',
                              'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'P_M*T': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'P_M*T',
                              'layout': {'windows': ['train', 'test'], 'id': 1}}
                }



            elif self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA':
                self.params_loss = ['Loss', 'Loss_ET', 'Loss_EA', 'Accuracy_EA', 'Accuracy_ET']  # Loss_EA
                self.log_loss.register(self.params_loss)

                # monitor training
                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss': {'dtype': 'running_mean'},
                    'Loss_EA': {'dtype': 'running_mean'},
                    'Loss_ET': {'dtype': 'running_mean'},
                    'Accuracy_EA': {'dtype': 'running_mean'},
                    'Accuracy_ET': {'dtype': 'running_mean'},
                }

                self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
                self.params_visualizer = {
                    'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss',
                             'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Loss_EA': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_EA',
                                'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Loss_ET': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_ET',
                                'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Accuracy_EA': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_EA',
                                    'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Accuracy_ET': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_ET',
                                    'layout': {'windows': ['train', 'test'], 'id': 1}},
                }




        else:
            if self.adverserial_type == 'OptNet':
                self.params_loss = ['Loss_Adversary', 'Loss_Target', 'Accuracy_Adversary', 'Accuracy_Target']
                self.log_loss.register(self.params_loss)

                # monitor testing
                self.monitor = plugins.Monitor()
                self.params_monitor = {
                    'Loss_Adversary': {'dtype': 'running_mean'},
                    'Loss_Target': {'dtype': 'running_mean'},
                    'Accuracy_Adversary': {'dtype': 'running_mean'},
                    'Accuracy_Target': {'dtype': 'running_mean'},
                }

                self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
                self.params_visualizer = {
                    'Loss_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Adversary',
                                    'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Accuracy_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Adversary',
                                        'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Loss_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Target',
                                    'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Accuracy_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Target',
                                        'layout': {'windows': ['train', 'test'], 'id': 1}},

                }

            elif self.adverserial_type == 'SGDA' or self.adverserial_type =='ExtraSGDA':
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
                self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
                self.params_visualizer = {
                    'Loss_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Adversary',
                                       'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Accuracy_Adversary': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Adversary',
                                           'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Loss_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Loss_Target',
                                    'layout': {'windows': ['train', 'test'], 'id': 1}},
                    'Accuracy_Target': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'Accuracy_Target',
                                        'layout': {'windows': ['train', 'test'], 'id': 1}},
                }


        self.monitor.register(self.params_monitor)
        self.visualizer.register(self.params_visualizer)

        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Test [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})'
            for item in self.params_loss:
                self.print_formatter += '|' + item + ' {:.4f}'

            self.print_formatter += '| lam: {:.2f}'
            self.print_formatter += '| adv_type: {:.10s}'
            self.print_formatter += '| train_enc: {:b}'


        self.evalmodules = []
        self.losses = {}

    def model_eval(self):
        self.model['Encoder'].eval()
        self.model['E-Adversary'].eval()
        self.model['E-Target'].eval()
        self.model['Adversary'].eval()
        self.model['Target'].eval()

    def test(self, epoch, dataloader, lam, reg):

        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        if self.log_type == 'progressbar':
            # progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<2}'.format('Test'), max=len(dataloader))

        end = time.time()

        for i, (inputs, labels, sensitives) in enumerate(dataloader):

            ############################
            # Evaluate Network
            ############################
            batch_size = inputs.size(0)


############################################ if T < A
            Y = torch.zeros(batch_size, self.total_classes).scatter_(1, labels.unsqueeze(1).long(), 1)
            S = torch.zeros(batch_size, self.total_classes).scatter_(1, sensitives.unsqueeze(1).long()+
                                                                     self.total_classes-self.nclasses_A, 1)

            self.inputs.resize_(inputs.size()).copy_(inputs)
            self.labels.resize_(labels.size()).copy_(labels)
            self.sensitives.resize_(sensitives.size()).copy_(sensitives)


            self.output_E= self.model['Encoder'](self.inputs)
            Y = Y.to(self.device)
            S = S.to(self.device)

            if self.encoder == True:

                if self.adverserial_type=='OptNet':

                    if self.args.loss_type_e == 'Projection':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.output_E, S, Y, lam, reg, self.device)
                    elif self.args.loss_type_e == 'Projection_poly':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.output_E, S, Y, lam, reg, self.device,
                                                                         self.args.c, self.args.d)
                    elif self.args.loss_type_e == 'Projection_gauss':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.output_E, S, Y, lam, reg, self.device,
                                                                         self.args.sigma)
                    elif self.args.loss_type_e == 'Projection_gauss_linear':
                        loss, loss_A, loss_T = self.criterion['Encoder'](self.output_E, S, Y, lam, reg, self.device,
                                                                         self.args.sigma)

                    self.model['Encoder'].zero_grad()

                    loss_A = loss_A.item()
                    loss_T = loss_T.item()
                    loss = loss.item()


                elif self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA':
                    output_A = self.model['E-Adversary'](self.output_E)
                    output_T = self.model['E-Target'](self.output_E)

                    loss_A = self.criterion['E-Discriminator'](output_A, S)  # outputs_A.squeeze()
                    loss_T = self.criterion['E-Target'](output_T, Y)  # outputs_T.squeeze()

                    loss = (1 - lam) * loss_T - lam * loss_A

                    self.model['Encoder'].zero_grad()
                    self.model['E-Adversary'].zero_grad()
                    self.model['E-Target'].zero_grad()
                    loss_A = loss_A.item()

                    loss_T = loss_T.item()

                    loss = loss.item()
                    # import pdb; pdb.set_trace()

            else:

                if self.adverserial_type == 'OptNet':
                    output_A = self.model['Adversary'](self.output_E)
                    output_T = self.model['Target'](self.output_E)

                    loss_A = self.criterion['Adversary'](output_A.squeeze(), S)
                    loss_T = self.criterion['Target'](output_T.squeeze(), Y)
                    self.model['Encoder'].zero_grad()
                    self.model['Adversary'].zero_grad()
                    self.model['Target'].zero_grad()

                    loss_A = loss_A.item()
                    loss_T = loss_T.item()

                elif self.adverserial_type =='SGDA' or self.adverserial_type == 'ExtraSGDA':
                    output_A = self.model['Adversary'](self.output_E)
                    output_T = self.model['Target'](self.output_E)

                    loss_A = self.criterion['Adversary'](output_A.squeeze(), S)
                    loss_T = self.criterion['Target'](output_T.squeeze(), Y)

                    loss_A = loss_A.item()
                    loss_T = loss_T.item()

            if self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA' or self.encoder == False:
                acc_A = self.evaluation['Adversary'](output_A, self.sensitives.long())
                acc_T = self.evaluation['Target'](output_T, self.labels.long())

                acc_A = acc_A.item()
                acc_T = acc_T.item()

            if self.encoder ==True:

                if self.adverserial_type=='OptNet':
                    self.losses['Loss'] = loss
                    self.losses['P_M*A'] = loss_A
                    self.losses['P_M*T'] = loss_T
                    self.monitor.update(self.losses, batch_size)

                elif self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA':
                    self.losses['Loss'] = loss
                    self.losses['Loss_ET'] = loss_T
                    self.losses['Loss_EA'] = loss_A
                    self.losses['Accuracy_EA'] = acc_A
                    self.losses['Accuracy_ET'] = acc_T
                    self.monitor.update(self.losses, batch_size)

                    nan = float('Nan')
                    if loss == nan or loss_A == nan or loss_T == nan:
                        import pdb; pdb.set_trace()

            else:
                if self.adverserial_type =='OptNet':

                    self.losses['Accuracy_Adversary'] = acc_A
                    self.losses['Accuracy_Target'] = acc_T
                    self.losses['Loss_Adversary'] = loss_A
                    self.losses['Loss_Target'] = loss_T

                    self.monitor.update(self.losses, batch_size)
                elif self.adverserial_type == 'SGDA' or self.adverserial_type == 'ExtraSGDA':
                    self.losses['Loss_Adversary'] = loss_A
                    self.losses['Accuracy_Adversary'] = acc_A
                    self.losses['Loss_Target'] = loss_T
                    self.losses['Accuracy_Target'] = acc_T

                    self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':

                processed_data_len += len(inputs)

                bar.suffix = self.print_formatter.format(
                    *[processed_data_len, len(dataloader.sampler)]
                     + [self.losses[key] for key in self.params_monitor]
                     + [lam]
                     + [self.adverserial_type]
                     + [self.encoder]
                )

                bar.next()
                end = time.time()
                bar.finish()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)


        self.visualizer.update(loss)

        return loss

