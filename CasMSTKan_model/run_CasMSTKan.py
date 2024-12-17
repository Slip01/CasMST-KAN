import numpy as np
import pickle
import config
from torch.utils.data import dataset
from torch.utils.data import DataLoader
import torch

import CasMST_KAN_model
import random
from tqdm.auto import tqdm
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('Model_size：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


class CasData(dataset.Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.id_train, self.x, self.L, self.y_train, self.sz_train, self.time_train, self.interval_popularity_train,self.vocabulary_size = pickle.load(open(self.root_dir, 'rb'))
        self.n_time_interval = config.n_time_interval
        self.n_steps = config.n_steps
    def __getitem__(self, idx):
        id = self.id_train[idx]

        y = self.y_train[idx]

        L = self.L[idx].todense()
        interval_popuplarity = self.interval_popularity_train[idx]
        time = self.time_train[idx]
        time = np.array(time,dtype=float)

        # temp = np.zeros(100)
        # for i in range(len(time)):
        #     temp[i] = time[i]
        # time = temp


        x = self.x[idx]
        x_ = self.x[idx].todense()
        x_ = torch.tensor(x_,dtype=torch.float32)
        time = torch.tensor(time,dtype=torch.float32)
        L = torch.tensor(L, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # time_interval_index_sample = torch.tensor(time_interval_index_sample, dtype=torch.float32)
        # rnn_index_temp = torch.tensor(rnn_index_temp, dtype=torch.float32)
        size = self.sz_train[idx]
        size = np.log2(size)
        size_train = torch.tensor(size,dtype=torch.float32)
        interval_popuplarity = torch.tensor(interval_popuplarity,dtype=torch.float32)

        return x_, L, time, y,size_train,interval_popuplarity
        # return x_,L,time,y,time_interval_index_sample,rnn_index_temp,

    def __len__(self):
        return len(self.sz_train)

# def main(args):
def main(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    ############################################
    Dataset = CasData(config.train_data)
    setup_seed(args.seed)
    learning_rate = 0.0015
    weight_decay = 5e-4
    # learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.001)
    # weight_decay = trial.suggest_uniform('weight_decay', 0.0001, 0.001)

    print("learning_rate = "+str(learning_rate))
    print("weight_decay = "+str(weight_decay))
    # train_dataset[5]
    # test_dataset = CasData(config.test_data)
    # val_dataset = CasData(config.val_data)
    test_size = int(len(Dataset)*0.15)
    val_size = int(len(Dataset)*0.15)
    train_size = int(len(Dataset)) - test_size - val_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(Dataset,
                                                                                  [train_size, test_size,
                                                                                   val_size],generator=torch.Generator().manual_seed(0))


    print("train_size:",train_size,"test_size:",test_size,"val_size:",val_size)
    train_dataloader = DataLoader(train_dataset,config.batch_size,drop_last=True)
    test_dataloader = DataLoader(test_dataset, config.batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, config.batch_size, drop_last=True)

    ############################################
    GCN_hidden_size=args.GCN_hidden_size
    GCN_hidden_size2 =args.GCN_hidden_size2
    MLP_hidden1 = args.MLP_hidden1
    MLP_hidden2 = args.MLP_hidden2
    Activation_fc = args.Activation_fc

    grid_size = 4
    spline_order = 3
    kernel_size = 5

    model = CasMST_KAN_model.CasMSTKanNet(input_dim=128, GCN_hidden_size=GCN_hidden_size,
                                          GCN_hidden_size2=GCN_hidden_size2, MLP_hidden1=MLP_hidden1, MLP_hidden2=MLP_hidden2, Activation_fc=Activation_fc,
                                          grid_size=grid_size, spline_order=spline_order,
                                          kernel_size=kernel_size,
                                          scale1=3, scale2=5)
    model = model.to(device)
    getModelSize(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)


    # model.load_state_dict(torch.load('./model.pth'))


    best_val_loss = 999999
    n_epochs = 40
    patience = 10
    epoch_no_improvement = 0

    for epoch in range(1,n_epochs+1):

        print("-------------Epoch:{}-----------".format(epoch))
        train_loss = []
        train_MAPE = []

        test_loss = []
        test_MAPE = []
        val_loss = []
        val_MAPE = []

        for data in train_dataloader:
            x, L,time,y,size_train,interval_popularity= data
            time = time.to(device)
            x = x.to(device)
            L = L.to(device)

            y = y.to(device)
            y = torch.reshape(y, [config.batch_size, 1])
            interval_popularity = interval_popularity.to(device)

            pred,nodes_pred = model(L, x,time,interval_popularity)
            optimizer.zero_grad()
            loss = torch.mean(torch.pow((pred - y), 2))

            error = torch.mean(torch.pow((pred - y), 2))
            MAPE_error = torch.mean(torch.abs(pred-y)/(torch.log2(torch.pow(2,y)+1)+torch.log2(torch.pow(2,pred)+1))*2)

            loss.backward()
            optimizer.step()

            train_loss.append(error.item())
            train_MAPE.append(MAPE_error.item())


        print("Train_MSLE：{:.3f},Train_MAPE:{:.3f}".format(np.mean(train_loss),np.mean(train_MAPE)))
        torch.cuda.empty_cache()
        with torch.no_grad():
            for data in val_dataloader:
                x, L,time, y,sz,interval_popularity= data

                time = time.to(device)
                x = x.to(device)
                L = L.to(device)

                y = y.to(device)
                y = torch.reshape(y, [config.batch_size, 1])
                interval_popularity = interval_popularity.to(device)

                pred,_ = model(L,x,time,interval_popularity)
                loss = torch.mean(torch.pow((pred - y), 2))
                MAPE_error = torch.mean(torch.abs(pred-y)/(torch.log2(torch.pow(2,y)+1)+torch.log2(torch.pow(2,pred)+1))*2)
                val_loss.append(loss.item())
                val_MAPE.append(MAPE_error.item())

        print("Val_MSLE:{:.3f},Val_MAPE:{:.3f}".format(np.mean(val_loss),np.mean(val_MAPE)))
        if (best_val_loss > np.mean(val_loss)):
            best_val_loss = np.mean(val_loss)
            epoch_no_improvement = 0
            print("Save best model")
            torch.save(model.state_dict(), 'model.pth')
        else:
            epoch_no_improvement += 1
            print("No improvement since epoch {}".format(epoch - epoch_no_improvement))
            if epoch_no_improvement >= patience:
                print("Early stopping!")
                break

    torch.cuda.empty_cache()
    model.load_state_dict(torch.load('./model.pth'))
    with torch.no_grad():
        for data in test_dataloader:
            x,L,time,y,sz,interval_popularity= data

            time = time.to(device)
            x = x.to(device)
            L = L.to(device)

            y = y.to(device)
            y = torch.reshape(y, [config.batch_size, 1])
            interval_popularity = interval_popularity.to(device)

            pred,nodes_pred = model(L,x,time,interval_popularity)
            loss = torch.mean(torch.pow((pred - y), 2))
            error = torch.mean(torch.pow((pred - y), 2))
            MAPE_error = torch.mean(torch.abs(pred-y)/(torch.log2(torch.pow(2,y)+1)+torch.log2(torch.pow(2,pred)+1))*2)
            torch.mean(torch.abs(pred-y)/torch.log2(torch.pow(2,y)+1))
            test_loss.append(error.item())
            test_MAPE.append(MAPE_error.item())

    print("-------------Final Test--------------")
    print("Test_MSLE:{:.3f},Test_MAPE:{:.3f}".format(np.mean(test_loss),np.mean(test_MAPE)))

    del model
    torch.cuda.empty_cache()

    return np.mean(test_loss)



if __name__ == '__main__':
    args = CasMST_KAN_model.get_params()

    # optuna
    # study = optuna.create_study()
    # study.optimize(main, n_trials=60)
    # print(study.best_trial)
    # print(study.best_params)
    # print(study.best_value)

    main(args)



