import os
from glob import glob
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from loaddata import Loaders
from networks import MaskCore
from easydict import EasyDict
from train_models import fix_seeds
from flags import args

def epoch_mask_train(
    model, 
    loader, 
    optimizer, 
    criterion,
    mask_decay, 
    schedule, 
    device
):

    model.train()
    model.core.eval()
    logs = EasyDict(
        invariance=torch.zeros(len(loader)), 
        sparsity=torch.zeros(len(loader))
    )

    for i, (inputs, targets, m_outs) in enumerate(loader):
        inputs = inputs.to(device)
        outs = model(inputs)
        
        invariance = criterion(outs, targets.to(device))
        invariance -= criterion(m_outs.to(device), targets.to(device))
        invariance = invariance**2
        logs.invariance[i] = invariance.item()
        invariance = torch.exp(invariance)

        norm = torch.norm(model.mask.weight, p=1)
        logs.sparsity[i] = norm
        sparsity = mask_decay * norm
        
        loss = invariance + sparsity
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if schedule:
            schedule.step()
            
        w = model.mask.weight.data
        w = w.clamp(0., 1.)
        model.mask.weight.data = w
    
    logs.invariance = logs.invariance.mean().item()
    logs.sparsity = logs.sparsity.mean().item()
    return logs

def epoch_mask_test(
    model, 
    loader, 
    device
):
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1).detach().cpu()
            correct += predicted.eq(targets.view_as(predicted)).sum()
    accuracy = 100 * correct.numpy()/len(loader.dataset)
    return accuracy

def train_mask(
    architecture, 
    pretrained_model,
    epochs,
    class_portion,
    lr,
    schedule,
    mask_size,
    mask_decay,
    patience, 
    seed,
    save_name
):
    
    # GPU configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    fix_seeds(seed)
    
    # load the model architecture
    model = MaskCore(
        mask_size=mask_size, 
        num_classes=5,
        architecture=architecture,
        pretrained_model=pretrained_model
    )

    # freeze the weights of the pre-trained network
    for p in model.core.parameters():
        p.requires_grad = False
    
    model = model.to(device)
    cudnn.benchmark = True
    
    # init lambda for sparsity loss term
    init_sparsity = torch.norm(model.mask.weight.data, p=1)
    mask_decay = mask_decay / init_sparsity
    
    # optimization configs
    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loaders = Loaders(
        batch_size=1, 
        class_portion=class_portion, 
        shuffle_train=True,
        shuffle_test=True,
        num_workers=3,
        transform='n',
        pretrained_model=pretrained_model,
        architecture=architecture
    )
    loader = loaders.testloader()
    init_accuracy = epoch_mask_test(model, loader, device) 
    
    # learning rate scheduling
    if schedule:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            epochs=epochs,
            steps_per_epoch=len(loader),
            pct_start=schedule
        )
    else: sched = False
    
    # prep logs
    logs = EasyDict(invariance=[], sparsity=[])
 
    # Train
    iterbar = tqdm(range(1, epochs + 1), total=epochs)
    cnt = 0
    
    for epoch in iterbar:
        losses = epoch_mask_train(
            model,
            loader,
            optimizer,
            lossfunction,
            mask_decay,
            sched,
            device
        )
        
        # get losses
        logs.invariance.append(losses.invariance)
        logs.sparsity.append(losses.sparsity/init_sparsity)
        accuracy = epoch_mask_test(model, loader, device)      
        
        description = f'Invariance: {logs.invariance[epoch-1]:.3f} - '\
                      f'Sparsity: {logs.sparsity[epoch-1]*100:.2f}% - '\
                      f'Init Accuracy: {init_accuracy:.2f}% - '\
                      f'Masked Accuracy: {accuracy:.2f}% - '\
                      f'Count: {cnt}'
        
        iterbar.set_description(desc=description)
        
        # stop at convergence
        if epoch > 1 and accuracy/init_accuracy >= 0.99: 
            torch.save(model.mask.weight.detach().cpu(), save_name)
            e = epoch-1
            if abs(logs.sparsity[e] - logs.sparsity[e-1]) <= 1e-5:
                cnt = cnt + 1
            else: cnt = 0
        if cnt==patience: 
            break
    del model
            
def main(
    architecture, 
    pretrained_model_folder, 
    mask_save_folder
):
    
    if not os.path.exists(mask_save_folder):
        os.makedirs(mask_save_folder)
        
    model_paths = glob(pretrained_model_folder + f'/{architecture}*.pt')
    if len(model_paths)==0: print('base models have not been trained yet')
    else:    
        for p in model_paths:
            pretrained_model = p
            p = os.path.basename(p).rsplit('_')[-1]
            save_path = mask_save_folder + f'/{architecture}_mask_' + p
            if not os.path.exists(save_path):
                train_mask(
                    architecture, 
                    pretrained_model,
                    epochs=args.mask_epochs,
                    class_portion=args.class_portion,
                    lr=args.mask_lr,
                    schedule=args.mask_schedule,
                    mask_size=args.img_size,
                    mask_decay=args.mask_decay,
                    patience=args.mask_patience,
                    seed=args.seed,
                    save_name=save_path
                )
            else: print(f'{save_path} is already trained')
                
if __name__ == "__main__":
    main(
        architecture=args.architecture, 
        pretrained_model_folder=args.model_save_folder, 
        mask_save_folder=args.mask_save_folder
    )
