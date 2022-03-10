import os
import numpy as np
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from loaddata import Loaders
from networks import MaskCore
from easydict import EasyDict
from attacks import PGD
from nnfunctions import fix_seeds
from flags import args

def model_predictions(
    architecture,
    pretrained_model,
    adv_epsilon,
    seed
):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    fix_seeds(seed)
    
    loader = Loaders(
        batch_size=1,
        class_portion=1.0,
        shuffle_train=False,
        shuffle_test=False,
        num_workers=0,
        transform='n',
        pretrained_model=None,
    ).testloader()
    
    # load the model architecture
    model = MaskCore(
        mask_size=(128, 128), 
        num_classes=5, 
        architecture=architecture,
        pretrained_model=pretrained_model
    )
    
    # send model to device
    model = model.to(device)
    cudnn.benchmark = True
    atk = PGD(model, eps=adv_epsilon, alpha=adv_epsilon/8, steps=10)

    images = torch.zeros(len(loader.dataset), 1, 1, 128, 128)
    model_outs = torch.zeros(len(loader.dataset), 1, 5)
    predictions = torch.zeros(len(loader.dataset)).long()
    true_targets = torch.zeros(len(loader.dataset)).long()
    
    model.eval()
    for i, (inputs, targets) in enumerate(loader):
        if adv_epsilon > 0.:
            inputs = atk(inputs, targets)

        images[i] = inputs
        inputs = inputs.to(device)
        outputs = model(inputs)
        model_outs[i] = outputs.detach().cpu()
        predictions[i] = torch.argmax(outputs, dim=1).detach().cpu().item()
        true_targets[i] = targets.item()
    
    match = predictions.eq(true_targets)
    predictions = predictions.view(len(loader.dataset), 1)
    return EasyDict(
        images=images, 
        predictions=predictions, 
        model_outs=model_outs, 
        match=match
    )

def get_single_img_data(architecture, pretrained_model_folder, num_images=None):
    print("Building Standard Model Data...")
    clean = model_predictions(
        architecture=architecture,
        pretrained_model=pretrained_model_folder + f'/{architecture}_basemodel_n.pt', 
        adv_epsilon=0., 
        seed=args.seed
    )
    print("Building Adversarially Trained Model Data...")
    adv_trained = model_predictions(
        architecture=architecture,
        pretrained_model=pretrained_model_folder + f'/{architecture}_basemodel_adv.pt', 
        adv_epsilon=0., 
        seed=args.seed
    )
    print("Building Adversarially Attacked Model Data...")
    adv_attacked = model_predictions(
        architecture=architecture,
        pretrained_model=pretrained_model_folder + f'/{architecture}_basemodel_n.pt', 
        adv_epsilon=args.adv_eps, 
        seed=args.seed
    )

    cadv = torch.where(clean.match==adv_trained.match)[0]
    catk = torch.where(clean.match==~adv_attacked.match)[0]
    idx = np.intersect1d(cadv, catk)
    
    if num_images and len(idx) > num_images:
        print("Collecting Restriction on num_images choice")
        idx = np.random.choice(idx, size=num_images, replace=False)
    
    return EasyDict(
        idx=idx, 
        images=clean.images[idx],
        predictions=clean.predictions[idx],
        standard_model_outs=clean.model_outs[idx],
        adv_trained_model_outs=adv_trained.model_outs[idx],
        adv_images=adv_attacked.images[idx],
        adv_predictions=adv_attacked.predictions[idx],
        adv_attacked_model_outs=adv_attacked.model_outs[idx]
        )

def get_single_img_pred(model, img, device):
    model.eval()
    with torch.no_grad():
        output = model(img.to(device))
        prediction = torch.argmax(output, dim=1).detach().cpu().item()
    return prediction

def epoch_single_img_mask_train(
    model, 
    img,
    target,
    m_out,
    optimizer, 
    criterion,
    mask_decay, 
    schedule, 
    device
):

    model.train()
    model.core.eval()
    logs = EasyDict()

    img = img.to(device)
    out = model(img)

    invariance = criterion(out, target.to(device))
    invariance -= criterion(m_out.to(device), target.to(device))
    invariance = invariance**2
    logs.invariance = invariance.item()
    invariance = torch.exp(invariance)

    norm = torch.norm(model.mask.weight, p=1)
    logs.sparsity = norm.item()
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
    return logs

def train_single_img_mask(
    img, 
    target, 
    m_out,
    architecture,
    pretrained_model,
    epochs, 
    lr, 
    schedule,
    mask_size,
    mask_decay, 
    patience,
    seed
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
    
    # send model to device
    model = model.to(device)
    cudnn.benchmark = True
    
    # init lambda for sparsity loss term
    init_sparsity = torch.norm(model.mask.weight.data, p=1)
    mask_decay = mask_decay / init_sparsity
    
    # optimization configs
    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # get initial prediction
    init_pred = get_single_img_pred(model, img, device)
    
    # learning rate scheduling
    if schedule:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            epochs=epochs,
            steps_per_epoch=200,
            pct_start=schedule
        )
    else:
        sched = False
    
    # prep logs
    logs = EasyDict(invariance=[], sparsity=[])
 
    # Train
    iterbar = tqdm(range(1, epochs + 1), total=epochs)
    cnt = 0

    for epoch in iterbar:
        losses = epoch_single_img_mask_train(
            model, 
            img,
            target,
            m_out,
            optimizer, 
            lossfunction,
            mask_decay, 
            sched, 
            device
        )
        # print logs
        logs.invariance.append(losses.invariance)
        logs.sparsity.append(losses.sparsity/init_sparsity)
        pred = get_single_img_pred(model, img, device)
        
        description = f'Invariance: {logs.invariance[epoch-1]:.3f} - '\
                      f'Sparsity: {logs.sparsity[epoch-1]*100:.2f}% - '\
                      f'Initial Prediction: {init_pred} - '\
                      f'Final Prediction: {pred} - '\
                      f'Count: {cnt}'
        iterbar.set_description(desc=description)
        
        # stop at convergence
        if epoch > 1 and init_pred==pred:
            e = epoch-1
            dl = abs(logs.invariance[e] - logs.invariance[e-1])
            dn = abs(logs.sparsity[e] - logs.sparsity[e-1])
            if dl <= 1e-6 and dn <= 1e-5:
                cnt = cnt + 1
            else: 
                cnt = 0
        if cnt==patience: 
            break
    
    mask = model.mask.weight.detach().cpu()
    return mask, pred

def main(
    architecture, 
    pretrained_model_folder,
    mask_save_folder,
    num_images=None
):
    
    if not os.path.exists(mask_save_folder):
        os.makedirs(mask_save_folder)
    
    if not os.path.exists(mask_save_folder + f'/{architecture}_single_img_data.pt'):
        data = get_single_img_data(
            architecture=architecture, 
            pretrained_model_folder=pretrained_model_folder,
            num_images=num_images
        )
        print("Saving Single image Data...")
        torch.save(data, mask_save_folder + f'/{architecture}_single_img_data.pt')
    else:
        data = torch.load(
            mask_save_folder + f'/{architecture}_single_img_data.pt', 
            map_location='cpu'
        )
        if len(data.idx) != num_images:
            data = get_single_img_data(
                architecture=architecture, 
                pretrained_model_folder=pretrained_model_folder,
                num_images=num_images
            )
            print("Saving Single image Data...")
            torch.save(data, mask_save_folder + f'/{architecture}_single_img_data.pt')
    
    masks = torch.zeros(len(data.idx), 1, 1, *args.img_size)
    adv_trained_masks = torch.zeros(len(data.idx), 1, 1, *args.img_size)
    adv_attacked_masks = torch.zeros(len(data.idx), 1, 1, *args.img_size)
    
    final_predictions = torch.zeros(len(data.idx)).long()
    adv_trained_final_predictions = torch.zeros(len(data.idx)).long()
    adv_attacked_final_predictions = torch.zeros(len(data.idx)).long()
    
    for i in range(len(data.idx)):
        print(f"Iteration {i+1}")
        masks[i], final_predictions[i] = train_single_img_mask(
            img=data.images[i], 
            target=data.predictions[i], 
            m_out=data.standard_model_outs[i], 
            architecture=architecture,
            pretrained_model=pretrained_model_folder + f'/{architecture}_basemodel_n.pt',
            epochs=args.single_img_mask_epochs,
            lr=args.single_img_mask_lr, 
            schedule=args.single_img_mask_schedule,
            mask_size=args.img_size,
            mask_decay=args.single_img_mask_decay, 
            patience=args.single_img_mask_patience, 
            seed=args.seed
        )
        # adv. trained mask
        adv_trained_masks[i], adv_trained_final_predictions[i] = train_single_img_mask(
            img=data.images[i], 
            target=data.predictions[i], 
            m_out=data.adv_trained_model_outs[i], 
            architecture=architecture,
            pretrained_model=pretrained_model_folder + f'/{architecture}_basemodel_adv.pt',
            epochs=args.single_img_mask_epochs,
            lr=args.single_img_mask_lr, 
            schedule=args.single_img_mask_schedule,
            mask_size=args.img_size,
            mask_decay=args.single_img_mask_decay, 
            patience=args.single_img_mask_patience, 
            seed=args.seed
        )
        # adv. attacked mask
        adv_attacked_masks[i], adv_attacked_final_predictions[i] = train_single_img_mask(
            img=data.adv_images[i], 
            target=data.adv_predictions[i], 
            m_out=data.adv_attacked_model_outs[i],
            architecture=architecture,
            pretrained_model=pretrained_model_folder + f'/{architecture}_basemodel_n.pt',
            epochs=args.single_img_mask_epochs,
            lr=args.single_img_mask_lr, 
            schedule=args.single_img_mask_schedule,
            mask_size=args.img_size,
            mask_decay=args.single_img_mask_decay, 
            patience=args.single_img_mask_patience, 
            seed=args.seed
        )
        
        single_img_masks = EasyDict(
            masks=masks, 
            adv_trained_masks=adv_trained_masks,
            adv_attacked_masks=adv_attacked_masks,
            final_predictions=final_predictions,
            adv_trained_final_predictions=adv_trained_final_predictions,
            adv_attacked_final_predictions=adv_attacked_final_predictions   
        )
            
    torch.save(single_img_masks, mask_save_folder + f'/{architecture}_single_img_masks.pt')

if __name__ == "__main__":
    main(
        architecture=args.architecture, 
        mask_save_folder=args.single_img_mask_save_folder, 
        num_images=args.num_images
    )



