import gc
import itertools
import random
from evaluation.evaluation import *
import numpy as np
# import optuna
import pandas as pd
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluation.evaluation import multiclass_acc
from models.attention import Dot_Attention
from transformers import optimization
from models.fake_news_model import FakeNewsModel, calculate_loss
from utils.utils import AvgMeter, print_lr, EarlyStopping, CheckpointSaving
from utils.ddp_utils import is_main_process, reduce_tensor, save_checkpoint


def batch_constructor(config, batch):
    b = {}
    for key, value in batch.items():
        if key != 'text':
            b[key] = value.to(config.device)
        else:
            b[key] = value
    return b


def report_simple_classification(truth, pred):
    """简化的分类报告，只显示关键指标"""
    from sklearn.metrics import classification_report
    import numpy as np
    
    truth = [i.cpu().numpy() for i in truth]
    pred = [i.cpu().numpy() for i in pred]

    pred = np.concatenate(pred, axis=0)
    truth = np.concatenate(truth, axis=0)

    report = classification_report(truth, pred, zero_division=0, output_dict=True)
    
    print(f"    Class 0 (Real):  Precision: {report['0']['precision']:.3f}, Recall: {report['0']['recall']:.3f}, F1: {report['0']['f1-score']:.3f}")
    print(f"    Class 1 (Fake):  Precision: {report['1']['precision']:.3f}, Recall: {report['1']['recall']:.3f}, F1: {report['1']['f1-score']:.3f}")
    print(f"    Overall Accuracy: {report['accuracy']:.3f}")

# ++++++++++++++++++++++++++++++++++++++++++++++++
def train_full_model_epoch(config, model, train_loader, optimizer, epoch, transformer_scheduler):
    loss_Dirichlet_meter = AvgMeter('train')
    warm_up_epoch = 1
    targets = []
    predictions = []
    
    # 只在主进程显示进度条
    from utils.ddp_utils import is_main_process
    from tqdm import tqdm
    
    if is_main_process():
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f'Epoch {epoch+1} Training', leave=False)
    else:
        pbar = enumerate(train_loader)
    
    for index, batch in pbar:
        batch = batch_constructor(config, batch)
        evidence, evidence_a, loss_Dirichlet, vt = model(batch, epoch)
        vt.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if epoch <= warm_up_epoch:
            transformer_scheduler.step()
        #print('lr', transformer_scheduler.get_last_lr())
        #print('loss', vt, '\n')
        optimizer.step()
        optimizer.zero_grad()

        count = batch["id"].size(0)
        loss_Dirichlet_meter.update(vt, count)
        target = batch['label'].detach()
        targets.append(target)
        _, output = torch.max(evidence.data, 1)
        prediction = output.detach()
        predictions.append(prediction)
        
        # 更新进度条
        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'Loss': f'{vt.item():.4f}',
                'Avg_Loss': f'{loss_Dirichlet_meter.avg:.4f}'
            })
    
    losses = loss_Dirichlet_meter
    return losses, targets, predictions

# ++++++++++++++++++++++++++++++++++++++++++++++++
def train_teacher_branch_epoch(config, model, train_loader, optimizer, scalar, epoch):
    loss_meter = AvgMeter('train')
    c_loss_meter = AvgMeter('train')
    vt_meter = AvgMeter('train')
    loss_Dirichlet_meter = AvgMeter('train')

    targets = []
    predictions = []
    
    # 只在主进程显示进度条
    from utils.ddp_utils import is_main_process
    from tqdm import tqdm
    
    if is_main_process():
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f'Teacher Branch Epoch {epoch+1}', leave=False)
    else:
        pbar = enumerate(train_loader)
    
    for index, batch in pbar:
        batch = batch_constructor(config, batch)
        evidence, evidence_a, loss_Dirichlet, vt = model(batch, epoch)
        loss, c_loss, s_loss = calculate_loss(model)

        vt.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        #if (index + 1) % 2:
        optimizer.step()
        optimizer.zero_grad()

        count = batch["id"].size(0)
        loss_meter.update(loss.detach(), count)
        vt_meter.update(vt.detach(), count)
        loss_Dirichlet_meter.update(loss_Dirichlet, count)

        target = batch['label'].detach()
        targets.append(target)
        _, output = torch.max(evidence.data, 1)

        prediction = output.detach()
        predictions.append(prediction)
        
        # 更新进度条
        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'Loss': f'{vt.item():.4f}',
                'Avg_Loss': f'{vt_meter.avg:.4f}'
            })

    losses = (vt_meter, loss_Dirichlet_meter)
    return losses, targets, predictions


def validation_epoch(config, model, validation_loader, epoch):
    loss_meter = AvgMeter('validation')
    s_loss_meter = AvgMeter('validation')
    total_loss_meter = AvgMeter('validation')
    loss_Dirichlet_meter = AvgMeter('validation')

    targets = []
    predictions = []

    predictions_a = []

    # 只在主进程显示进度条
    from utils.ddp_utils import is_main_process
    from tqdm import tqdm
    
    if is_main_process():
        pbar = tqdm(validation_loader, desc=f'Epoch {epoch+1} Validation', leave=False)
    else:
        pbar = validation_loader
    
    for batch in pbar:
        batch = batch_constructor(config, batch)
        with torch.no_grad():

            evidence, evidence_a, loss_Dirichlet, vt = model(batch, epoch)
            loss, c_loss, s_loss = calculate_loss(model)
            s_loss = vt
            total_loss = loss_Dirichlet
            count = batch["id"].size(0)
            loss_meter.update(loss.detach(), count)
            s_loss_meter.update(s_loss.detach(), count)
            loss_Dirichlet_meter.update(loss_Dirichlet.detach(), count)
            total_loss_meter.update(total_loss.detach(), count)

            _, output = torch.max(evidence.data, 1)
            _, output_a = torch.max(evidence_a.data, 1)

            prediction = output.detach()
            predictions.append(prediction)
            target = batch['label'].detach()
            targets.append(target)

            prediction_a = output_a.detach()
            predictions_a.append(prediction_a)

    losses = (s_loss_meter, total_loss_meter, loss_Dirichlet_meter)
    return losses, targets, predictions, predictions_a


def supervised_train(config, train_loader, validation_loader, rank=0, world_size=1, trial=None):
    torch.cuda.empty_cache()
    checkpoint_path2 = checkpoint_path = str(config.output_path) + '/checkpoint.pt'
    if trial:
        checkpoint_path2 = str(config.output_path) + '/checkpoint_' + str(trial.number) + '.pt'

    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    scalar = torch.cuda.amp.GradScaler()
    model = FakeNewsModel(config).to(config.device)
    
    # 包装模型为DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 获取模型参数 (处理DDP包装)
    model_params = model.module if isinstance(model, DDP) else model
    
    params = [
        {"params": model_params.image_encoder.parameters(),
         "lr": config.image_encoder_lr, "name": 'image_encoder'},
         {"params": model_params.text_encoder.parameters(),
          "lr": config.text_encoder_lr, "weight decay": 0.001, "name": 'text_encoder'},
        {"params": model_params.text_classifier.parameters(), "lr": 0.0003,
         "weight_decay": 0.0001,
         'name': 'text classifier'},
        {"params": model_params.image_classifier.parameters(), "lr": 0.0003,
         "weight_decay": 0.0001,
         'name': 'image classifier'},
        {"params": model_params.image_encoder_teacher.parameters(),
         "lr": config.image_encoder_teacher_lr, "weight_decay": 0.001, "name": 'image_encoder_teacher'},
        {"params": model_params.text_encoder_teacher.parameters(),
         "lr": config.text_encoder_teacher_lr, "weight_decay": 0.001,  "name": 'text_encoder_teacher'},
        {"params": itertools.chain(model_params.image_projection.parameters(), model_params.text_projection.parameters()),
         "lr": 0.0005, "weight_decay": 0.001, 'name': 'projection'},
        {"params": itertools.chain(model_params.attend_vt.parameters(), model_params.attend_tv.parameters()),
         "lr": 0.0005, "weight_decay": 0.001, 'name': 'attention'},
        {"params": model_params.classifier.parameters(), "lr": 0.0003,
         "weight_decay": 0.001,
         'name': 'classifier'}
    ]
    optimizer = torch.optim.AdamW(params, amsgrad=True)
    transformer_scheduler = optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                         num_warmup_steps=200,
                                                                         num_training_steps=5000,
                                                                         num_cycles=4,
                                                                         last_epoch=- 1
                                                                         )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,
                                                              eta_min=1e-6)
    params_teacher =  [
        {"params": model_params.image_encoder_teacher.parameters(), "lr": config.image_encoder_teacher_lr, "name": 'image_encoder_teacher'},
        {"params": model_params.text_encoder_teacher.parameters(), "lr": config.text_encoder_teacher_lr, "name": 'text_encoder_teacher'},
        {"params": itertools.chain(model_params.image_projection.parameters(), model_params.text_projection.parameters()),
         "lr": 0.0005, "weight_decay": config.head_weight_decay, 'name': 'projection'},
        {"params": model_params.classifier.parameters(), "lr": 0.0003,
         "weight_decay": 0.0001,
         'name': 'classifier'}
    ]
    optimizer_teacher = torch.optim.AdamW(params_teacher, amsgrad=True)

    early_stopping = EarlyStopping(patience=config.patience, delta=config.delta, path=checkpoint_path, verbose=True)
    checkpoint_saving = CheckpointSaving(path=checkpoint_path, verbose=True)

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []

    validation_accuracy, validation_loss = 0, 1
    
    # 第一阶段：固定训练指定数量的epoch
    first_stage_epochs = config.first_stage_epochs
    for epoch in range(first_stage_epochs):
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"FULL MODEL TRAINING - EPOCH {epoch + 1}/{first_stage_epochs}")
            print(f"{'='*60}")
        
        model.train()
        Dirichlet_loss, train_truth, train_predict= train_full_model_epoch(config, model, train_loader, optimizer, epoch, transformer_scheduler)
        train_accuracy = multiclass_acc(train_truth, train_predict)
        
        # 从AvgMeter对象中提取平均损失值
        if hasattr(Dirichlet_loss, 'avg'):
            loss_value = Dirichlet_loss.avg
        else:
            loss_value = Dirichlet_loss
        
        # 在多卡训练中规约指标
        if world_size > 1:
            train_accuracy = reduce_tensor(torch.tensor(train_accuracy).cuda()).item()
            loss_value = reduce_tensor(torch.tensor(loss_value).cuda()).item()
        
        if is_main_process():
            print(f"Train - Acc: {train_accuracy:.4f}, Loss: {loss_value:.4f}")

        model.eval()
        with torch.no_grad():
            test_loss, test_truth, test_pred, test_predi_a = validation_epoch(config, model, validation_loader, epoch=epoch)
            test_acc = multiclass_acc(test_truth, test_pred)
            
            # 从AvgMeter对象中提取平均损失值
            if isinstance(test_loss, tuple):
                # test_loss是(s_loss_meter, total_loss_meter, loss_Dirichlet_meter)的元组
                test_loss_values = tuple(loss.avg if hasattr(loss, 'avg') else loss for loss in test_loss)
            else:
                test_loss_values = test_loss.avg if hasattr(test_loss, 'avg') else test_loss
            
            # 在多卡训练中规约验证指标
            if world_size > 1:
                test_acc = reduce_tensor(torch.tensor(test_acc).cuda()).item()
                if isinstance(test_loss_values, tuple):
                    test_loss_values = tuple(reduce_tensor(torch.tensor(loss).cuda()).item() for loss in test_loss_values)
                else:
                    test_loss_values = reduce_tensor(torch.tensor(test_loss_values).cuda()).item()
            
            if is_main_process():
                print(f"Valid - Acc: {test_acc:.4f}, Loss: {test_loss_values[0] if isinstance(test_loss_values, tuple) else test_loss_values:.4f}")
                
                # 简化的分类报告 - 只在每10个epoch或最后一个epoch显示
                if epoch % 10 == 0 or epoch == first_stage_epochs - 1:
                    print(f"Detailed Classification Report:")
                    report_simple_classification(test_truth, test_pred)
        if epoch > 1:
            lr_scheduler.step()
            if is_main_process():
                print_lr(optimizer)
        # 第一阶段不使用early stopping，只保存最佳模型
        if checkpoint_saving:
            checkpoint_saving(test_acc, model)


    if is_main_process():
        print('Full Model Training Over~')
        print('\n' + '='*60)
        print(f'Starting Teacher Branch Fine-tuning with Early Stopping (max {config.second_stage_epochs} epochs)')
        print('='*60)
    
    # 重置Early Stopping和Checkpoint Saving，因为这是一个新的训练阶段
    if early_stopping:
        early_stopping.counter = 0
        early_stopping.best_score = None
        early_stopping.early_stop = False
        early_stopping.val_loss_min = np.Inf
        if is_main_process():
            print("Early stopping enabled for Teacher Branch Fine-tuning")
    
    if checkpoint_saving:
        checkpoint_saving.best_score = None
        checkpoint_saving.val_acc_max = 0
        if is_main_process():
            print("Checkpoint saving reset for Teacher Branch Fine-tuning")
    
    # 第二阶段：最多训练指定数量的epoch，但会使用early stopping
    second_stage_epochs = config.second_stage_epochs
    for epoch in range(second_stage_epochs):
        if is_main_process():
            print(f'Teacher Branch Fine-tuning - Epoch: {epoch + 1}/{second_stage_epochs}')
        gc.collect()

        model.train()
        train_loss, train_truth, train_pred = train_teacher_branch_epoch(config, model, train_loader, optimizer_teacher, scalar, epoch=epoch)

        model.eval()
        with torch.no_grad():
            validation_loss, validation_truth, validation_pred, validation_pred_a = validation_epoch(config, model, validation_loader, epoch = epoch)
            validation_accuracy = multiclass_acc(validation_truth, validation_pred)
            
            # 在多卡训练中规约验证指标
            if world_size > 1:
                validation_accuracy = reduce_tensor(torch.tensor(validation_accuracy).cuda()).item()
                # 由于validation_loss是tuple，需要特别处理
                if isinstance(validation_loss, tuple):
                    validation_loss_values = tuple(reduce_tensor(torch.tensor(loss.avg if hasattr(loss, 'avg') else loss).cuda()).item() for loss in validation_loss)
                else:
                    validation_loss_values = reduce_tensor(torch.tensor(validation_loss.avg if hasattr(validation_loss, 'avg') else validation_loss).cuda()).item()
            else:
                # 单卡情况下，提取avg值
                if isinstance(validation_loss, tuple):
                    validation_loss_values = tuple(loss.avg if hasattr(loss, 'avg') else loss for loss in validation_loss)
                else:
                    validation_loss_values = validation_loss.avg if hasattr(validation_loss, 'avg') else validation_loss
            
            if is_main_process():
                print('Validation================')
                print(f'Valid - Acc: {validation_accuracy:.4f}, Loss: {validation_loss_values[0] if isinstance(validation_loss_values, tuple) else validation_loss_values:.4f}')
                print()

        train_accuracy = multiclass_acc(train_truth, train_pred)

        if is_main_process():
            print('Train================')
            print(f'Train - Acc: {train_accuracy:.4f}, Loss: {train_loss[0].avg if hasattr(train_loss[0], "avg") else train_loss[0]:.4f}')
            print()
            print_lr(optimizer)
            print_lr(optimizer_teacher)

        if early_stopping:
            # 使用规约后的损失值
            early_stopping_loss = validation_loss_values[0] if isinstance(validation_loss_values, tuple) else validation_loss_values
            early_stopping(early_stopping_loss, model)
            if early_stopping.early_stop:
                if is_main_process():
                    print("Early stopping")
                break
        if checkpoint_saving:
            checkpoint_saving(validation_accuracy, model)

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        validation_accuracies.append(validation_accuracy)
        validation_losses.append(validation_loss)

        if trial:
            trial.report(validation_accuracy, epoch)
            if trial.should_prune():
                print('trial pruned')
                # raise optuna.exceptions.TrialPruned()

        print()

    if checkpoint_saving:
        model = FakeNewsModel(config).to(config.device)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        with torch.no_grad():
            validation_loss, validation_truth, validation_pred, validation_pred_a = validation_epoch(config, model, validation_loader, epoch=epoch)
        validation_accuracy = multiclass_acc(validation_pred, validation_truth)
        if trial and validation_accuracy >= config.wanted_accuracy:
            loss_accuracy = pd.DataFrame(
                {'train_loss': train_losses, 'train_accuracy': train_accuracies, 'validation_loss': validation_losses,
                 'validation_accuracy': validation_accuracies})
            torch.save({'model_state_dict': model.state_dict(),
                        'parameters': str(config),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_accuracy': loss_accuracy}, checkpoint_path2)

    if not checkpoint_saving:
        loss_accuracy = pd.DataFrame(
            {'train_loss': train_losses, 'train_accuracy': train_accuracies, 'validation_loss': validation_losses,
             'validation_accuracy': validation_accuracies})
        torch.save(model.state_dict(), checkpoint_path)
        if trial and validation_accuracy >= config.wanted_accuracy:
            torch.save({'model_state_dict': model.state_dict(),
                        'parameters': str(config),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_accuracy': loss_accuracy}, checkpoint_path2)
    return validation_accuracy

