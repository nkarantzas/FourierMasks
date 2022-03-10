import os
from glob import glob
import numpy as np
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from loaddata import Loaders
from networks import Core, MaskCore
from easydict import EasyDict
from attacks import FGSM, PGD
from flags import args

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def epoch_train(
    model, 
    loader, 
    optimizer, 
    criterion, 
    schedule,
    attack,
    device
):
    
    model.train()
    logs = EasyDict(loss=torch.zeros(len(loader)))
    correct = 0
    
    for i, (inputs, targets) in enumerate(loader):
        
        if attack:
            inputs = attack(inputs, targets)
        
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        logs.loss[i] = loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if schedule:
            schedule.step()
        predicted = torch.argmax(outputs, dim=1).detach().cpu()
        correct += predicted.eq(targets.view_as(predicted)).sum()
        
    logs.loss = logs.loss.mean().item()
    accuracy = 100.*correct/len(loader.dataset)
    logs.accuracy = accuracy.numpy()
    
    return logs

def epoch_test(
    model, 
    loader, 
    criterion, 
    device
):
    
    model.eval()
    logs = EasyDict(loss=torch.zeros(len(loader)))
    correct = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.to(device))
            logs.loss[i] = loss.item()
            predicted = torch.argmax(outputs, dim=1).detach().cpu()
            correct += predicted.eq(targets.view_as(predicted)).sum()
    
    logs.loss = logs.loss.mean().item()
    accuracy = 100.*correct/len(loader.dataset)
    logs.accuracy = accuracy.numpy()
    
    return logs

def train_model(
    architecture,
    epochs, 
    batch_size,
    lr, 
    schedule=None,
    weight_decay=0.,
    seed=31,
    savename=None,
    transform='n',
    adv_eps=0.
):
    
    # GPU configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    fix_seeds(seed)

    # get the data
    loaders = Loaders(
        batch_size=batch_size, 
        class_portion=1.,
        shuffle_train=True,
        shuffle_test=False,
        num_workers=3,
        transform=transform,
        pretrained_model=None
    )
    
    trainloader = loaders.trainloader()
    testloader = loaders.testloader()
    
    # get model
    model = Core(num_classes=5, architecture=architecture)
    model = model.to(device)
    cudnn.benchmark = True
    
    # optimization configs
    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    # learning rate scheduling
    if schedule:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            lr, 
            epochs=epochs,
            steps_per_epoch=len(trainloader),
            pct_start=schedule
        )
    else: sched = False
    
    if adv_eps > 0.: 
        train_attack = PGD(
            model, 
            eps=adv_eps, 
            alpha=adv_eps/8, 
            steps=10
        )
        
    # prep results
    logs = EasyDict(
        train_loss=[], 
        test_loss=[], 
        train_accuracy=[], 
        test_accuracy=[]
    )

    # Train
    iterbar = tqdm(range(1, epochs + 1), total=epochs)
    for epoch in iterbar:
        
        trainlogs = epoch_train(
            model, 
            trainloader, 
            optimizer, 
            lossfunction,
            sched,
            train_attack if adv_eps > 0. else None,
            device
        )
        testlogs = epoch_test(
            model, 
            testloader,
            lossfunction,
            device
        )
        logs.train_loss.append(trainlogs.loss)
        logs.test_loss.append(testlogs.loss)
        logs.train_accuracy.append(trainlogs.accuracy)
        logs.test_accuracy.append(testlogs.accuracy)
        
        description = f'Train Loss: {trainlogs.loss:.2f} - '\
                      f'Test Loss: {testlogs.loss:.2f} - '\
                      f'Train Acc: {trainlogs.accuracy:.2f}% - '\
                      f'Test Acc: {testlogs.accuracy:.2f}%'
        
        iterbar.set_description(desc=description)
        if logs.test_accuracy[-1] >= max(logs.test_accuracy) and savename:
            logs.state_dict = model.state_dict()
            torch.save(logs, savename)
    del model

def main(architecture, model_save_folder):
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    transforms = ['n', 'adv', 'sn', 'tn', 'rn']
    for t in transforms:
        save_path = model_save_folder + f'/{architecture}_basemodel_{t}.pt'
        if not os.path.exists(save_path):
            train_model(
                architecture, 
                epochs=args.model_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                schedule=args.schedule,
                weight_decay=args.weight_decay,
                seed=args.seed,
                savename=save_path,
                transform='n' if t=='adv' else t,
                adv_eps=args.adv_eps if t=='adv' else 0.
            )
        else: print(f'{save_path} is already trained')
            
if __name__ == "__main__":
    main(architecture=args.architecture, model_save_folder=args.model_save_folder)