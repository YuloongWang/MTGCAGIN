import os
import util
import random
import torch
import torch.utils.data
import numpy as np
from model_mt import *
from dataset import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
torch.autograd.set_detect_anomaly(True)


def topk_loss(s, ratio, minibatch_size=None):
    if ratio > 0.5:
        ratio = 1 - ratio
    res = 0
    for i, r in enumerate(s):
        s[r] = torch.sigmoid(s[r]).view(minibatch_size, -1)
        s[r] = s[r].sort(dim=1).values
        res += -torch.log(s[r][:, -int(s[r].size(1) * ratio):] + 1e-10).mean() - torch.log(1 - s[r][:, :int(s[r].size(1) * ratio)] + 1e-10).mean()
    return res


def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, roi, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho, pool_weight, score = model(dyn_v, dyn_a, t, sampling_endpoints, roi)
    # loss = criterion(logit, label.to(device))
    # reg_ortho *= reg_lambda
    # loss += reg_ortho

    loss = criterion(logit, label.to(device))
    loss_p = (torch.norm(pool_weight, p=2) - 1) ** 2  # pool1_weight_loss
    # loss_tpk = topk_loss(score, 0.8, dyn_v[roi[0]].size(0))
    loss = loss + reg_lambda * (reg_ortho + loss_p)

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return logit, loss, attention, latent, reg_ortho


def train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    dataset = Datasetinsomia(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, dynamic_length=argv.dynamic_length)
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=0, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    logger_train = util.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)
    logger_test = util.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)
    logger_save = util.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)

    # start experiment
    for k in range(checkpoint['fold'], argv.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)
        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout)
        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss()
        lossall = 0

        # define optimizer and learning rate scheduler
        # optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr, eps=1e-4, weight_decay=0.0001)
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, steps_per_epoch=len(dataloader_train), pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_test = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
        # summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )

        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            # print('=============================')
            logger_train.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate_train = 0.0
            reg_ortho_accumulate_train = 0.0
            for i, x in enumerate(tqdm(dataloader_train, ncols=60, desc=f'train k:{k} e:{epoch}')):
                # process input data
                dyn_a_all = {}
                if i==0: dyn_v_all = {}
                t_all = {}
                for j, r in enumerate(argv.roi):
                    dyn_a, sampling_points = util.bold.process_dynamic_fc(x[f'timeseries_{r}'], argv.window_size, argv.window_stride, argv.dynamic_length)
                    sampling_endpoints = [p+argv.window_size for p in sampling_points]
                    if i==0 or (len(dyn_a)<argv.minibatch_size): dyn_v = repeat(torch.eye(dataset.num_nodes[j]), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                    # if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                    t = x[f'timeseries_{r}'].permute(1, 0, 2)
                    dyn_a_all[r] = dyn_a.to(device)
                    if i==0 or (len(dyn_a) < argv.minibatch_size): dyn_v_all[r] = dyn_v[:len(dyn_a)].to(device)
                    t_all[r] = t.to(device)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v_all,
                    dyn_a=dyn_a_all,
                    sampling_endpoints=sampling_endpoints,
                    t=t_all,
                    label=label,
                    roi=argv.roi,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler)
                # logit = logit - torch.max(logit)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                if torch.isnan(prob).any():
                    print(logit)
                    exit(0)
                loss_accumulate_train += loss.detach().cpu().numpy()
                reg_ortho_accumulate_train += reg_ortho.detach().cpu().numpy()
                logger_train.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader_train))

            # summarize results
            samples_train = logger_train.get(k)
            metrics_train = logger_train.evaluate(k)
            lossall_train = loss_accumulate_train / len(dataloader_train)
            # print(lossall_train)
            summary_writer.add_scalar('loss', loss_accumulate_train/len(dataloader_train), epoch)
            summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate_train/len(dataloader_train), epoch)
            summary_writer.add_pr_curve('precision-recall', samples_train['true'], samples_train['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics_train.items() if not key=='fold']
            [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer.flush()
            print(lossall_train, metrics_train)

            # test()
            # define logging objects
            fold_attention = {'node_attention': [], 'time_attention': []}

            logger_test.initialize(k)
            # logger_save.initialize(k)
            dataset.set_fold(k, train=False)
            loss_accumulate_test = 0.0
            reg_ortho_accumulate_test = 0.0
            latent_accumulate = []
            # for i, x in enumerate(tqdm(dataloader_test, ncols=60, desc=f'test k:{k} e:{epoch}')):
            for i, x in enumerate(dataloader_test):
                with torch.no_grad():
                    # process input data
                    dyn_a_all = {}
                    if i==0: dyn_v_all = {}
                    t_all = {}
                    for j, r in enumerate(argv.roi):
                        dyn_a, sampling_points = util.bold.process_dynamic_fc(x[f'timeseries_{r}'], argv.window_size, argv.window_stride)
                        sampling_endpoints = [p + argv.window_size for p in sampling_points]
                        if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes[j]), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                        # if not dyn_v.shape[1] == dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes[j]), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                        # if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                        t = x[f'timeseries_{r}'].permute(1, 0, 2)
                        dyn_a_all[r] = dyn_a.to(device)
                        if i==0: dyn_v_all[r] = dyn_v[:len(dyn_a)].to(device)
                        t_all[r] = t.to(device)
                    label = x['label']

                    logit, loss, attention, latent, reg_ortho = step(
                        model=model,
                        criterion=criterion,
                        dyn_v=dyn_v_all,
                        dyn_a=dyn_a_all,
                        sampling_endpoints=sampling_endpoints,
                        t=t_all,
                        label=label,
                        roi=argv.roi,
                        reg_lambda=argv.reg_lambda,
                        clip_grad=argv.clip_grad,
                        device=device,
                        optimizer=None,
                        scheduler=None)
                    pred = logit.argmax(1)
                    prob = logit.softmax(1)
                    logger_test.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                    loss_accumulate_test += loss.detach().cpu().numpy()
                    reg_ortho_accumulate_test += reg_ortho.detach().cpu().numpy()

                    fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                    fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                    latent_accumulate.append(latent.detach().cpu().numpy())

            # summarize results
            samples_test = logger_test.get(k)
            metrics_test = logger_test.evaluate(k)
            lossall_test = loss_accumulate_test / len(dataloader_test)
            # print(loss_accumulate_test / len(dataloader_test))
            summary_writer_test.add_scalar('loss', loss_accumulate_test / len(dataloader_test), epoch)
            summary_writer_test.add_scalar('reg_ortho', reg_ortho_accumulate_test / len(dataloader_test), epoch)
            summary_writer_test.add_pr_curve('precision-recall', samples_test['true'], samples_test['prob'][:, 1], epoch)
            [summary_writer_test.add_scalar(key, value, epoch) for key, value in metrics_test.items() if not key == 'fold']
            [summary_writer_test.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer_test.flush()
            print('test: ', lossall_test, metrics_test)

            if epoch == 0: lossall = 0
            if metrics_test['accuracy'] > lossall and epoch > argv.num_epochs-2:
                lossall = metrics_test['accuracy']
                print('Save best model')
                torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))

                # finalize fold
                logger_save.samples[k] = logger_test.samples[k]
                # logger_save.to_csv(argv.targetdir, k)
                [np.save(os.path.join(argv.targetdir, 'attention', str(k), f'{key}.npy'), np.concatenate(value)) for key, value in fold_attention.items()]
                np.save(os.path.join(argv.targetdir, 'attention', str(k), 'latent.npy'), np.concatenate(latent_accumulate))
                del fold_attention

            # save checkpoint
            torch.save({
                'fold': k,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(argv.targetdir, 'checkpoint.pth'))

        # finalize fold
        logger_save.to_csv(argv.targetdir, k)
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    # summary_writer_val.close()
    os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))

    # finalize experiment
    logger_save.to_csv(argv.targetdir)
    final_metrics = logger_save.evaluate()
    print('====')
    print(final_metrics)
    summary_writer.close()
    summary_writer_test.close()
    torch.save(logger_save.get(), os.path.join(argv.targetdir, 'samples.pkl'))

