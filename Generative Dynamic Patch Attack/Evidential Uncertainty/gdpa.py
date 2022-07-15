import torch
import torch.nn.functional as F
import torchvision
import time
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_log_writer, scale_theta, scale_pattern
from models import load_generator, load_model_vggface
# from data import load_vggface_unnormalized, normalize_vggface, load_imagenet_unnormalize, normalize_imagenet, normalize_imagenette, load_imagenette_unnormalize
from data import load_vggface_unnormalized, normalize_vggface
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# A simple adversarial patch fault detection
# visualization of working
# python gdpa.py --dataset vggface --data_path Data --vgg_model_path new_ori_model.pt --patch_size 71

def relu_evidence(y):
    return F.relu(y)

def print_confmat(y_pred,y_true, name):
    # constant for classes
    classes = (
                "a_j__buckley", 
                "a_r__rahman",
                "aamir_khan" ,
                "aaron_staton",
                "aaron_tveit",
                "aaron_yoo",
                "abbie_cornish",
                "abel_ferrara",
                "abigail_breslin",
                "abigail_spencer"
                )

    # classes = (
    #             "0" ,
    #             "1",
    #             "2",
    #             "3",
    #             "4",
    #             "5",
    #             "6",
    #             "7",
    #             "8", 
    #             "9",
    #             "10",
    #             "11",
    #             "13",
    #             "14",
    #             "15",
    #             "16", )

    # classes = (
    #             "ball" ,
    #             "building",
    #             "chainsaw",
    #             "dog",
    #             "fish",
    #             "mellophone",
    #             "parachute",
    #             "pump_station",
    #             "radio", 
    #             "truck",)

    # Build confusion matrix
    # print(y_true)
    # print(y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("confusion matrix")
    plt.savefig(name+'_confusion_matrix_output.png')
    plt.clf()


# def print_cf_mat(model,dataloaders,name,normalize_func):
#     y_pred = []
#     y_true = []
#     y_pred_distribution = []

#     # iterate over test data
#     # for inputs, labels in dataloaders:
#     for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders)):
#             output = model(normalize_func(inputs)) # Feed Network

#             evidence = relu_evidence(output)
#             alpha = evidence + 1
#             uncertainty = 10 / torch.sum(alpha, dim=1, keepdim=True)
#             y_pred_distribution.extend(uncertainty.detach().numpy().flatten())

#             output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#             y_pred.extend(output) # Save Prediction
            


#             labels = labels.data.cpu().numpy()
#             y_true.extend(labels) # Save Truth


    # print_cfmat_confmat(y_pred,y_true,name)

def cal_Ytrue_Ypres(model,inputs,labels,name):
    y_pred = []
    y_true = []
    y_pred_distribution=[]

    # for inputs, labels in dataloaders:
    output = model(inputs) # Feed Network
    evidence = relu_evidence(output)
    alpha = evidence + 1
    uncertainty = 10 / torch.sum(alpha, dim=1, keepdim=True)
    y_pred_distribution.extend(uncertainty.detach().numpy().flatten())

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

    return y_true,y_pred,y_pred_distribution 


def flatten(t):
    return [item for sublist in t for item in sublist]

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def move_m_p(aff_theta, pattern_s, device, alpha=1):
    bs = pattern_s.size()[0]
    image_with_patch = torch.zeros(bs, 3, 224, 224, device=device)
    mask_with_patch = torch.zeros(bs, 1, 224, 224, device=device)
    start = 111 - pattern_s.size()[2] // 2
    end = start + pattern_s.size()[2]
    image_with_patch[:, :, start:end, start:end] = pattern_s
    mask_with_patch[:, :, start:end, start:end] = alpha
    rot_theta = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).unsqueeze(0).to(device).repeat(bs, 1, 1)
    theta_batch = torch.cat((rot_theta, aff_theta.unsqueeze(2)), 2)
    grid = F.affine_grid(theta_batch, image_with_patch.size(), align_corners=True)
    pattern_s = F.grid_sample(image_with_patch, grid, align_corners=True)
    mask_s = F.grid_sample(mask_with_patch, grid, align_corners=True)
    return mask_s, pattern_s


def perturb_image(inputs, mp_generator, devide_theta, device, alpha=1, p_scale=10000):
    mask_generated, pattern_generated, aff_theta = mp_generator(inputs)
    aff_theta = scale_theta(aff_theta, devide_theta)
    pattern_s = scale_pattern(pattern_generated, p_scale=p_scale)
    mask_s, pattern_s = move_m_p(aff_theta, pattern_s, device, alpha=alpha)
    inputs = inputs * (1 - mask_s) + pattern_s * mask_s
    inputs = inputs.clamp(0, 1)
    return inputs


def train_gen_batch(inputs, targets, model, mp_generator, optimizer_gen, criterion,
                    loss_l_gen, devide_theta, normalize_func, device, alpha=1, p_scale=10000):
    mp_generator.train()
    model.eval()

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, device, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_func(inputs))
    loss = -criterion(outputs, targets)
    loss.backward()
    optimizer_gen.step()
    loss_l_gen.append(loss.cpu().detach().numpy())
    _, predicted = outputs.max(1)

    y_true,y_pred, y_pred_dist = cal_Ytrue_Ypres(model, normalize_func(inputs), targets, "FaceData_train")

    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs,y_true, y_pred, y_pred_dist


def test_gen_batch(inputs, targets, model, mp_generator,
                   optimizer_gen, devide_theta, normalize_func, device, alpha=1, p_scale=10000):
    mp_generator.eval()
    model.eval()

    inputs, targets = inputs.to(device), targets.to(device)
    optimizer_gen.zero_grad()
    inputs = perturb_image(inputs, mp_generator, devide_theta, device, alpha=alpha, p_scale=p_scale)
    outputs = model(normalize_func(inputs))
    _, predicted = outputs.max(1)

    y_true,y_pred, y_pred_dist2 = cal_Ytrue_Ypres(model, normalize_func(inputs), targets, "FaceData_test")

    return (~predicted.eq(targets)).sum().item(), targets.size(0), inputs, y_true, y_pred, y_pred_dist2


def gdpa(dataloader, dataloader_val, model, mp_generator, optimizer_gen, scheduler, criterion,
          epochs, devide_theta, alpha, normalize_func, writer, device):

    # out_y_true = []
    # out_y_pred = []
    # out_y_pred_dist1 = []

    # out_y_true2 = []
    # out_y_pred2 = []
    # out_y_pred_dist2 = []

    for epoch in range(epochs):
        start_time = time.time()
        print('epoch: {}'.format(epoch))
        # training
        loss_l_gen = []
        correct_gen = 0
        total_gen = 0
        y_true = []
        y_pred = []
        y_pred_dist1 = []

        y_true2 = []
        y_pred2 = []
        y_pred_dist2 = []

        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            y_tru=[]
            y_prd=[]
            y_prd_dis1 = []
            correct_batch, total_batch, final_ims_gen,y_tru, y_prd, y_prd_dis1 = train_gen_batch(inputs, targets, model,
                                                                        mp_generator,
                                                                        optimizer_gen, criterion,
                                                                        loss_l_gen, devide_theta, normalize_func,
                                                                        device, alpha=alpha, p_scale=10000)
            correct_gen += correct_batch
            total_gen += total_batch
            y_true.extend(y_tru)
            y_pred.extend(y_prd)
            y_pred_dist1.extend(y_prd_dis1)
        # training log
        loss = np.array(loss_l_gen).mean()
        asr = correct_gen / total_gen
        writer.add_scalar('train_gen/loss', loss, epoch)
        writer.add_scalar('train_gen/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_gen/{}'.format(epoch), final_ims_gen, epoch)
        # testing
        correct_gen2 = 0
        total_gen2 = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader_val)):
            y_tru2=[]
            y_prd2=[]
            y_prd_dis2 = []
            correct_batch, total_batch, final_ims_gen, y_tru2, y_prd2, y_prd_dis2 = test_gen_batch(inputs, targets, model,
                                                                       mp_generator,
                                                                       optimizer_gen, devide_theta, normalize_func,
                                                                       device, alpha=alpha, p_scale=10000)
            correct_gen2 += correct_batch
            total_gen2 += total_batch
            y_true2.extend(y_tru2)
            y_pred2.extend(y_prd2)
            y_pred_dist2.extend(y_prd_dis2)
        # testing log
        asr = correct_gen2 / total_gen2
        writer.add_scalar('test_gen/asr', asr, epoch)
        final_ims_gen = torchvision.utils.make_grid(final_ims_gen)
        writer.add_image('final_im_test/{}'.format(epoch), final_ims_gen, epoch)

        # out_y_true = y_true
        # out_y_pred = y_pred
        # out_y_pred_dist1 = y_pred_dist1

        # out_y_true2 = y_true2
        # out_y_pred2 = y_pred2
        # out_y_pred_dist2 = y_pred_dist2

        # scheduler
        scheduler.step()
        # time
        end_time = time.time()
        print(end_time - start_time)

    print_confmat(y_pred,y_true,"FaceData_traingen")
    print_confmat(y_pred2,y_true2,"FaceData_testgen")
    return y_pred_dist1,y_pred_dist2


def get_args():
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='gdpa')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default='vggface')
    parser.add_argument('--data_path', type=str, default='/home/xli62/uap/phattacks/glass/Data')
    parser.add_argument('--vgg_model_path', type=str,
                        default='/home/xli62/uap/phattacks/glass/donemodel/new_ori_model.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_gen', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args


def before_attack(model,dataloader_val,name,normalize_func):
    y_pred = []
    y_true = []
    y_pred_distribution = []

    # iterate over test data
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader_val)):
            y_tru2=[]
            y_prd2=[]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(normalize_func(inputs))
            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            uncertainty = 10 / torch.sum(alpha, dim=1, keepdim=True)
            y_pred_distribution.extend(uncertainty.detach().numpy().flatten())

            _, predicted = torch.max(model(normalize_func(inputs)), 1)

            y_pred.extend(predicted.data.cpu().numpy()) # Save Prediction

            labels = targets.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # for inputs, labels in dataloaders:
    #         output = model(inputs) # Feed Network

    #         evidence = relu_evidence(output)
    #         alpha = evidence + 1
    #         uncertainty = 10 / torch.sum(alpha, dim=1, keepdim=True)
    #         y_pred_distribution.extend(uncertainty.detach().numpy().flatten())

    #         _, predicted = torch.max(model(inputs), 1)

    #         y_pred.extend(predicted.data.cpu().numpy()) # Save Prediction

    #         labels = labels.data.cpu().numpy()
    #         y_true.extend(labels) # Save Truth


    # constant for classes
    classes = (
                "a_j__buckley", 
                "a_r__rahman",
                "aamir_khan" ,
                "aaron_staton",
                "aaron_tveit",
                "aaron_yoo",
                "abbie_cornish",
                "abel_ferrara",
                "abigail_breslin",
                "abigail_spencer"
                )

    # classes = (
    #             "0" ,
    #             "1",
    #             "2",
    #             "3",
    #             "4",
    #             "5",
    #             "6",
    #             "7",
    #             "8", 
    #             "9",
    #             "10",
    #             "11",
    #             "13",
    #             "14",
    #             "15",
    #             "16", )

    # classes = (
    #             "ball" ,
    #             "building",
    #             "chainsaw",
    #             "dog",
    #             "fish",
    #             "mellophone",
    #             "parachute",
    #             "pump_station",
    #             "radio", 
    #             "truck",)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.savefig(name+'_confusion_matrix.png')
    plt.clf()

    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(y_pred_distribution)
    plt.title("confidence plot")
    plt.ylabel('confidence level')
    # plt.xlabel('confidence level')
    plt.savefig(name+'y_pred_distribution_singlePlot.png')
    plt.clf()

    data1 = pd.DataFrame({0: y_pred_distribution})
    data = np.hstack((data1,data1))
    df = pd.DataFrame(data,columns=['validation','validation'])
    a=plt.ylabel('confidence level')
    boxplot = df.boxplot(grid=True)
    plt.title("confidence plot")
    plt.savefig(name+'_confidence_plot.png')
    plt.clf()

    data2 = pd.DataFrame({0: y_pred})
    data = np.hstack((data2,data2))
    df = pd.DataFrame(data,columns=['validation','validation'])
    a=plt.ylabel('confidence level')
    boxplot = df.boxplot(grid=True)
    plt.title("confidence plot")
    plt.savefig(name+'_confidence_plot_y_pred.png')
    plt.clf()

    return y_pred_distribution

def plot_confidence_matrix(y_dist, y_dist2,name):
    data1 = pd.DataFrame({0: y_dist})
    # data2 = pd.DataFrame({0: y_dist1})
    data3 = pd.DataFrame({0: y_dist2})
    data = np.hstack((data1,data3))
    # data = np.hstack((data,data3))
    df = pd.DataFrame(data,columns=['before_attack','testing'])
    a=plt.ylabel('confidence level')
    boxplot = df.boxplot(grid=True)
    plt.title("confidence plot")
    plt.savefig(name+'_confidence_plot.png')
    plt.clf()

    # data2 = pd.DataFrame({0: y_pred})
    # data = np.hstack((data2,data2))
    # df = pd.DataFrame(data,columns=['validation','validation'])
    # a=plt.ylabel('confidence level')
    # boxplot = df.boxplot(grid=True)
    # plt.title("confidence plot")
    # plt.savefig(name+'_confidence_plot_y_pred.png')
    # plt.clf()

def main():
    args = get_args()
    para = {'exp': args.exp, 'beta': args.beta, 'lr_gen': args.lr_gen,
            'epochs': args.epochs, 'alpha': args.alpha, 'patch_size': args.patch_size, 'dataset': args.dataset}
    writer, base_dir = get_log_writer(para)
    # data
    if para['dataset'] == 'vggface':
        dataloader, dataloader_val = load_vggface_unnormalized(args.batch_size, args.data_path)
        normalize_func = normalize_vggface
    elif para['dataset'] == 'imagenet':
        dataloader, dataloader_val = load_imagenet_unnormalize(args.batch_size, args.data_path)
        normalize_func = normalize_imagenet
    elif para['dataset'] == 'imagenette':
        dataloader, dataloader_val = load_imagenette_unnormalize(args.batch_size, args.data_path)
        normalize_func = normalize_imagenette

    # clf model
    if para['dataset'] == 'vggface':
        model_train = load_model_vggface(args.vgg_model_path)
    elif para['dataset'] == 'imagenet':
        model_train = models.vgg19(pretrained=True)
    elif para['dataset'] == 'imagenette':
        model_train = load_model_vggface(args.vgg_model_path)    

    model_train = model_train.to(args.device)
    model_train.eval()
    # gen model
    mp_generator = load_generator(para['patch_size'], 3, 64).to(args.device)
    # training setting
    optimizer_gen = torch.optim.Adam([
        {'params': mp_generator.parameters(), 'lr': para['lr_gen']}
    ], lr=0.1, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=50, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()

    y_pred_dist = before_attack(model_train,dataloader_val,"faceData_before_attack_val",normalize_func)

    # train and test
    y_pred_dist1, y_pred_dist2 = gdpa(dataloader, dataloader_val, model_train, mp_generator, optimizer_gen, scheduler,
          criterion, para['epochs'], para['beta'], para['alpha'], normalize_func, writer, args.device)

    print(len(y_pred_dist))
    print(len(y_pred_dist2))

    plot_confidence_matrix(y_pred_dist, y_pred_dist2,"FaceData_confidence_plot")

if __name__ == '__main__':
    main()
