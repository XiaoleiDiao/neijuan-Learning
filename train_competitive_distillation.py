import time

import torch
import wandb
import argparse
import os
import utils
from tqdm import tqdm
import torch.nn.functional as F
import csv
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser()

    '''
    Models choose：resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202. mobilenetV1, inceptionV1
    '''
    parser.add_argument("--model", type=str, default="resnet32", required=False)
    parser.add_argument("--dataset", type=str, default='cifar100', required=False)
    parser.add_argument("--gpu", type=str, default="0,1", required=False)
    parser.add_argument("--num_classes", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--optim", type=str, default='adamw', required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--lrscheduler", type=str, default='reduce', required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0005, required=False)
    parser.add_argument("--momentum", type=float, default=0.9, required=False)
    parser.add_argument("--gamma", type=float, default=0.1, required=False)
    parser.add_argument("--device", type=str, default='cuda', required=False)
    parser.add_argument("--topk", type=int, default=1, required=False)
    parser.add_argument("--eval_flag", type=bool, default=True, required=False)

    args = parser.parse_args()
    args.model = 'resnet32, resnet32, resnet32'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    return args


def train_one_epoch(epoch,
                    models,
                    train_dataloader,
                    optimizers,
                    criterion_ce,
                    criterion_kl,
                    train_losses,
                    args,
                    is_augmentation=False
                    ):

    with tqdm(total=len(train_dataloader), desc=f'{epoch}/{args.epochs}', postfix=dict, maxinterval=1.0) as pbar:

        for i_batch, (images, labels) in enumerate(train_dataloader):

            if is_augmentation:
                images, labels_B, lam = utils.mixed_image(images, labels)
                labels_B = labels_B.cuda()

            images = images.cuda()
            labels = labels.cuda()
            outputs = []

            # Calculate the output of each model:
            for m, model in enumerate(models):
                optimizers[m].zero_grad()
                outputs.append(model(images))

            # Calculate the loss of each model:
            if is_augmentation:
                ce_losses = [lam * criterion_ce(outputs[i], labels) + (1 - lam) * criterion_ce(outputs[i], labels_B) for i in range(len(models))]
            else:
                ce_losses = [criterion_ce(outputs[i], labels) for i in range(len(models))]

            # ----------------------
            # 1. competitive distillation, learn for the best-performing model
            # ----------------------
            for i in range(len(models)):
		    # Calculate the kl_loss of each model, if the ce_loss of the model is larger than the ce_loss of the traversed model,
                # then it will be calculated in kl_loss

                ce_loss = ce_losses[i]
                kl_loss = 0.
                n_kl = 0. # Record the number of involution learning models, used for averaging
                lr = utils.get_lr(optimizers[i])

               # 1. Compare models with each other, and learn from all models that perform better than themselves
                for j in range(len(models)):
                    if i != j and ce_loss > ce_losses[j]:
                        n_kl += 1
                        kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(Variable(outputs[j]), dim=1))
                if n_kl != 0:
                    loss = ce_loss + kl_loss / (n_kl)
                else:
                    loss = ce_loss

                # measure current metric
                _, predict = outputs[i].topk(args.topk, 1, True, True)
                predict = predict.t()
                if is_augmentation:
                    acc = (lam * predict.eq(labels.data).cpu().sum().float()
                           + (1 - lam) * predict.eq(labels_B.data).cpu().sum().float())
                else:
                    acc = predict.eq(labels.view_as(predict)).float().sum()

                loss.backward()
                optimizers[i].step()

                # save the loss of the model
                train_losses[i].update(ce_loss.item(), kl_loss.item() if kl_loss!=0 else kl_loss, loss.item(), acc.item(), epoch)

                # Show the loss for the current training epoch
                pbar.set_postfix({
                    # 'model': i,
                    # 'ce_loss': train_losses[i].ce_loss / (i_batch + 1),
                    # 'kl_loss': train_losses[i].kl_loss / (i_batch + 1),
                    'loss': train_losses[i].loss / (i_batch + 1),
                    'correct': train_losses[i].acc / len(train_dataloader.dataset),
                    'lr': lr,
                })

            # ----------------------
            # 2. competitive distillation: four models, learn for the first two best-performing models.
            # ----------------------
            # top_k1 = -1
            # top_k2 = -1
            # list_ce = sorted(ce_losses, reverse=False)
            # for i in range(4):
            #     if ce_losses[i] == list_ce[0]:
            #         top_k1 = i
            #     elif ce_losses[i] == list_ce[1]:
            #         top_k2 = i
            #
            # for i in range(len(models)):
            #     kl_loss = 0.
            #     ce_loss = ce_losses[i]
            #     loss = 0.
            #
            #     if i == top_k1:
            #         # kl_loss += criterion_kl(F.log_softmax(outputs[top_k1], dim=1),
            #         #                         F.softmax(outputs[top_k2].detach(), dim=1))
            #         # kl_loss += criterion_kl(F.log_softmax(outputs[top_k2].detach(), dim=1),
            #         #                         F.softmax(outputs[top_k1], dim=1))
            #         loss = ce_losses[top_k1] #+ kl_loss / 2
            #     elif i == top_k2:
            #         # kl_loss += criterion_kl(F.log_softmax(outputs[top_k2], dim=1),
            #         #                         F.softmax(outputs[top_k1].detach(), dim=1))
            #         # kl_loss += criterion_kl(F.log_softmax(outputs[top_k1].detach(), dim=1),
            #         #                         F.softmax(outputs[top_k2], dim=1))
            #         loss = ce_losses[top_k2] #+ kl_loss / 2
            #     else:
            #         kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
            #                                 F.softmax(outputs[top_k1].detach(), dim=1))
            #         kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
            #                                 F.softmax(outputs[top_k2].detach(), dim=1))
            #         loss = ce_losses[i] + kl_loss / 2
            #
            #     # measure current metric
            #     _, predict = outputs[i].topk(args.topk, 1, True, True)
            #     predict = predict.t()
            #     if is_augmentation:
            #         acc = (lam * predict.eq(labels.data).cpu().sum().float()
            #                + (1 - lam) * predict.eq(labels_B.data).cpu().sum().float())
            #     else:
            #         acc = predict.eq(labels.view_as(predict)).float().sum()
            #
            #     loss.backward()
            #     optimizers[i].step()
            #
            #     # record the loss
            #     train_losses[i].update(ce_loss.item(), kl_loss.item() if kl_loss != 0. else kl_loss, loss.item(),
            #                            acc.item(), epoch)
            #
            #     # show the loss in the current step
            #     pbar.set_postfix({
            #         # 'model': i,
            #         # 'ce_loss': train_losses[i].ce_loss / (i_batch + 1),
            #         # 'kl_loss': train_losses[i].kl_loss / (i_batch + 1),
            #         'loss': train_losses[i].loss / (i_batch + 1),
            #         'correct': train_losses[i].acc / len(train_dataloader.dataset),
            #         # 'lr': lr,
            #     })

            pbar.update(1)


def val(epoch, models, val_dataloader, criterion_ce, criterion_kl, val_losses, args, ):

    with torch.no_grad():
        for i in range(len(models)):
            models[i].eval()

        with tqdm(total=len(val_dataloader), desc=f'Epoch [{epoch}/{args.epochs}]') as pbar:
            for i_batch, (images, labels) in enumerate(val_dataloader):

                images = images.cuda()
                labels = labels.cuda()
                outputs = []

                # calculate the output of each model
                for model in models:
                    outputs.append(model(images))

                # ----------------------
                # 1.
                # ----------------------
                # calculate the loss of each model
                for i in range(len(models)):
                    ce_loss = criterion_ce(outputs[i], labels)
                    kl_loss = 0.

                   
                    for j in range(len(models)):
                        if i != j:
                            kl_loss += criterion_kl(F.log_softmax(outputs[i], dim=1),
                                                    F.softmax(outputs[j].detach(), dim=1))

                    loss = ce_loss + kl_loss / (len(models) - 1)

                    # measure current metric
                    _, predict = outputs[i].topk(args.topk, 1, True, True)
                    predict = predict.t()
                    acc = predict.eq(labels.view_as(predict)).float().sum()

                    # record the loss of models
                    val_losses[i].update(ce_loss.item(), kl_loss.item(), loss.item(), acc.item(), epoch)

             
                pbar.update(1)


def train(args, model_str, project_name, SAVE_PATH, is_augmentation=False):
    utils.set_random_seed(args.seed)
    args.model = model_str
    wandb.init(project='mutual_involution_learning', name=project_name)
    wandb.config.update(args, allow_val_change=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

    model_names, \
    models, \
    train_dataloader, \
    val_dataloader = utils.get_model_and_dataset(args)

    optimizers = utils.get_optimizer(args, models, len(models))
    schedulers = utils.get_scheduler(args, optimizers)

    n_train_batch = len(train_dataloader)
    n_train_data = len(train_dataloader.dataset)
    n_val_batch = len(val_dataloader)
    n_val_data = len(val_dataloader.dataset)
    best_val_acc = [0.] * len(models)
    best_train_acc = [0.] * len(models)

    utils.show_config(args)

    # training with multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f'use {torch.cuda.device_count()} GPUs')
        device = torch.device('cuda:0')
        new_models = []
        for k in range(len(models)):
            model0 = torch.nn.DataParallel(models[k], device_ids=[0,1]).cuda()
            new_models.append(model0)
        models = new_models

    for epoch in range(args.epochs):
        train_losses = []
        val_losses = []

        for i in range(len(models)):
            train_losses.append(utils.Totoal_Meter(n_train_batch, n_train_data))
            val_losses.append(utils.Totoal_Meter(n_val_batch, n_val_data))
            schedulers[i].step(epoch)
            models[i] = models[i].train()

        # train
        train_one_epoch(epoch, models, train_dataloader, optimizers,
                              criterion_ce, criterion_kl, train_losses, args, is_augmentation)

        # val
        if args.eval_flag:
            val(epoch, models, val_dataloader,
                      criterion_ce, criterion_kl, val_losses, args)

        # save and print
        # print(f'Epoch: [{epoch}/{args.epochs}]')
        for i in range(len(models)):
            train_res = train_losses[i].get_avg()
            val_res = val_losses[i].get_avg()

            if val_res['acc'] > best_val_acc[i]:
                best_val_acc[i] = val_res['acc']
            if train_res['acc'] > best_train_acc[i]:
                best_train_acc[i] = train_res['acc']
                # torch.save(models[i].state_dict(), os.path.join('result', model_names[i] + '_best.pth'))

            # print(
            #     f"{i}: {model_names[i]} "
            #     f" [train_ce_loss={train_res['ce_loss']:.4f} train_kl_loss={train_res['kl_loss']:.4f} train_loss={train_res['loss']:.4f} train_acc={train_res['acc']:.4f}]"
            #     f" [val_ce_loss={val_res['ce_loss']:.4f} val_kl_loss={val_res['kl_loss']:.4f} val_loss={val_res['loss']:.4f} val_acc={val_res['acc']:.4f}]"
            # )

            # save training information in wandb
            wandb.log({
                "{}_{}_{}_{}_ceLoss".format(i, model_names[i], args.dataset, 'train'): train_res['ce_loss'],
                "{}_{}_{}_{}_klLoss".format(i, model_names[i], args.dataset, 'train'): train_res['kl_loss'],
                "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'train'): train_res['loss'],
                "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'train'): train_res['acc'],
                "{}_{}_{}_{}_ceLoss".format(i, model_names[i], args.dataset, 'val'): val_res['ce_loss'],
                "{}_{}_{}_{}_klLoss".format(i, model_names[i], args.dataset, 'val'): val_res['kl_loss'],
                "{}_{}_{}_{}_loss".format(i, model_names[i], args.dataset, 'val'): val_res['loss'],
                "{}_{}_{}_{}_acc".format(i, model_names[i], args.dataset, 'val'): val_res['acc'],
                # 'jiaohu': train_res['jiaohu']
            }, step=epoch)

    with open(os.path.join(SAVE_PATH, 'involution_result.csv'), 'w', encoding='utf-8', newline='') as f:
        fileName = ['model_name', 'train_acc', 'val_acc']
        writer = csv.DictWriter(f, fieldnames=fileName)
        writer.writeheader()
        for i in range(len(model_names)):
            writer.writerow({'model_name':model_names[i],
                             'train_acc': best_train_acc[i],
                             'val_acc': best_val_acc[i],})

    wandb.finish()

def involution_(model_str, proj_name, SAVE_PATH, is_augmentation):
    args = parse_args()
    train(args, model_str, proj_name, SAVE_PATH, is_augmentation=is_augmentation)

if __name__ == '__main__':
    involution_()




