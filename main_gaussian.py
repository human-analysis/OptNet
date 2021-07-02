# main.py

import sys
import traceback
import torch
import random
import config
import utils
from model import Model
from train_gaussian import Trainer
from test_gaussian import Tester

# from dataloader import Dataloader
from dataloader import PrivacyDataLoader
from checkpoints import Checkpoints


def main():

    # parse the arguments
    args = config.parse_args()
    if (args.ngpu > 0 and torch.cuda.is_available()):
        device = "cuda:0"
    else:
        device = "cpu"


    alpha = args.alpha
    args.device = torch.device(device)

    m = 0
    for k in range(args.niters):

        dataloader1 = PrivacyDataLoader(args, encoder=1)
        dataloader2 = PrivacyDataLoader(args, encoder=0)
        for lam in alpha:
            if args.save_results:
                utils.saveargs(args)

            random.seed(args.manual_seed + k)
            torch.manual_seed(args.manual_seed + k)

            # initialize the checkpoint class

            checkpoints = Checkpoints(args)

            # Create Model
            models = Model(args)

            model, criterion, evaluation = models.setup(checkpoints)

            loaders_train1 = dataloader1.create("Train")
            loaders_train2 = dataloader2.create("Train")
            loaders_test = dataloader2.create("Test")

            trainer_train = Trainer (args, model, criterion, evaluation, lam, k, encoder=True)
            tester_train = Tester (args, model, criterion, evaluation, lam, k, encoder=True)
            trainer_test = Trainer (args, model, criterion, evaluation, lam, k, encoder=False)
            tester_test = Tester (args, model, criterion, evaluation, lam, k, encoder=False)


            loss_best = 1e10
            step = 0
            for epoch in range(int(args.nepochs_e)):

                lam1 = lam
                print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs_e))

                loss_train_E = trainer_train.train(epoch, loaders_train1, lam1, args.reg)

                with torch.no_grad():
                    loss_test_E = tester_train.test(epoch, loaders_test, lam1, args.reg)

                if loss_best > loss_test_E['Loss']:
                    loss_best = loss_test_E['Loss']
                    if args.save_results:
                        checkpoints.save(k, lam, 'Encoder', model['Encoder'])

            loss_best_t = 1e10
            loss_best_a = 1e10
            for epoch in range(int(args.nepochs)):
                # print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))
                loss_train = trainer_test.train(epoch, loaders_train2, lam, args.reg)
                # A_train[m] = loss_train['Accuracy_Adversary']
                # T_train[m] = loss_train['Accuracy_Target']
                with torch.no_grad():
                    loss_test = tester_test.test(epoch, loaders_test, lam, args.reg)

                if loss_best_a > loss_test['Loss_Adversary']:
                    loss_best_a = loss_test['Loss_Adversary']
                    if args.save_results:
                        checkpoints.save(k, lam, 'Adv', model['Adversary'])

                if loss_best_t > loss_test['Loss_Target']:
                    loss_best_t = loss_test['Loss_Target']
                    if args.save_results:
                        checkpoints.save(k, lam, 'tgt', model['Target'])



#

if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()
