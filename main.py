import syft as sf
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
import torch as th
from torch import nn, optim
import Model
import shap
from jacobian import JacobianReg
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
from torch.utils.data import TensorDataset, DataLoader



def create_workers(hook):
    workers=[]
    for i in range(5):
        workers.append(sf.VirtualWorker(hook, id="w"+str(i)))

    return workers


def le_change(data,columns):
    data_scaler = MinMaxScaler()
    data = pd.DataFrame(data_scaler.fit_transform(data), columns=columns)

    return data

def get_final(train, test):
    rivsed_train = pd.get_dummies(train)
    rivsed_test = pd.get_dummies(test)

    final_test, final_train = rivsed_test.align(rivsed_train, join='inner', axis=1)

    return final_test, final_train


def generate(train_fname,test_fname, workers, batch_size):
    train = loadarff(train_fname)
    test = loadarff(test_fname)

    train = pd.DataFrame(train[0])
    test = pd.DataFrame(test[0])

    train['service'] = train['service'].str.decode('utf-8')
    train['protocol_type'] = train['protocol_type'].str.decode('utf-8')
    train['flag'] = train['flag'].str.decode('utf-8')
    train['class'] = train['class'].str.decode('utf-8')

    test['service'] = test['service'].str.decode('utf-8')
    test['protocol_type'] = test['protocol_type'].str.decode('utf-8')
    test['flag'] = test['flag'].str.decode('utf-8')
    test['class'] = test['class'].str.decode('utf-8')

    train["class"]=np.where(train["class"]=="normal",0,1)
    test["class"] = np.where(test["class"] == "normal", 0, 1)

    train_class = train["class"]
    train = train.drop(["class"], axis=1)

    test_class = test["class"]
    test = test.drop(["class"], axis=1)

    final_train = pd.get_dummies(train)
    final_test = pd.get_dummies(test)

    final_train, final_test = final_train.align(final_test, join='inner', axis=1)

    columns = final_train.columns

    final_train = le_change(final_train, columns)
    final_test = le_change(final_test, columns)

    feat_name = final_train.columns
    train = np.array(final_train)
    test = np.array(final_test)

    verify, verify_class = train[-10:,:], train_class[-10:]

    train_loader = TensorDataset(th.tensor(train[:-10,:], dtype=th.float), th.tensor(train_class[:-10], dtype=th.long))
    test_loader = TensorDataset(th.tensor(test, dtype=th.float), th.tensor(test_class, dtype=th.long))

    train_loader = sf.FederatedDataLoader(train_loader.federate(workers), batch_size=batch_size, shuffle=True)
    test_loader = th.utils.data.DataLoader(test_loader, batch_size=batch_size)


    return train_loader, test_loader, verify, verify_class, feat_name



def create_model_optim(workers,lr):
    models=[]
    optims=[]
    regs=[]

    for i in workers[:-1]:
        model= Model.model(120,64).send(i)
        optim_for_model= th.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.3)
        reg = JacobianReg().send(i)
        models.append(model)
        optims.append(optim_for_model)
        regs.append(reg)

    return models, optims, regs




def encrypt_share_gradient(models,workers):
    for model in models:
        model.fc1.weight.data = model.fc1.weight.get().fix_prec().share(*workers)
        model.fc1.bias.data = model.fc1.bias.get().fix_prec().share(*workers)
        model.fc2.weight.data = model.fc2.weight.get().fix_prec().share(*workers)
        model.fc2.bias.data = model.fc2.bias.get().fix_prec().share(*workers)

    return models


def aggregate_gradient(models,workers):
    fc1_weight = list()
    fc1_bias = list()
    fc2_weight = list()
    fc2_bias = list()



    for model in models:
        fc1_weight.append(model.fc1.weight.data.clone().get())
        fc1_bias.append(model.fc1.bias.data.clone().get())
        fc2_weight.append(model.fc2.weight.data.clone().get())
        fc2_bias.append(model.fc2.bias.data.clone().get())


    model=Model.model(120,64)
    with th.no_grad():
        model.fc1.weight.set_((sum(fc1_weight) / len(fc1_weight)).float_prec())
        model.fc1.bias.set_((sum(fc1_bias) / len(fc1_bias)).float_prec())
        model.fc2.weight.set_((sum(fc2_weight) / len(fc2_weight)).float_prec())
        model.fc2.bias.set_((sum(fc2_bias) / len(fc2_bias)).float_prec())

    return model



def explain(pre_model,worker,verify,i):
    model = Model.model(120, 64)

    with th.no_grad():
      model.fc1.weight.set_(pre_model.fc1.weight.get())
      model.fc1.bias.set_(pre_model.fc1.bias.get())
      model.fc2.weight.set_(pre_model.fc2.weight.get())
      model.fc2.bias.set_(pre_model.fc2.bias.get())


    torch_data = th.from_numpy(verify).to(device).float()
    # explainer_shap = shap.GradientExplainer(model, torch_data)
    #
    # shap_values = explainer_shap.shap_values(torch_data)

    shap.initjs()
    #shap.plots.force(shap_values[0])
    #shap.summary_plot(shap_values[0], torch_data.numpy(), feature_names=feat_name, plot_size=(13, 10), show=True)


    explainer_shap = shap.DeepExplainer(model, torch_data)

    shap_values = explainer_shap.shap_values(torch_data)


    shap.initjs()
    shaps=shap.force_plot(explainer_shap.expected_value[0],shap_values[1][0,:], torch_data.detach().numpy()[0,:], feature_names=feat_name)
    shap.save_html("instance_pos.html",shaps)

    shapss=shap.force_plot(explainer_shap.expected_value[0], shap_values[0][0, :], torch_data.detach().numpy()[0, :],
                    feature_names=feat_name)
    shap.save_html("instance_neg.html", shapss)

    shap.summary_plot(shap_values[0], torch_data.numpy(), feature_names=feat_name, plot_size=(13, 10), show=True)

    model=model.send(worker)

    return model

    # shap.force_plot(shap_values[0],verify,"1")



def train_federated_model(workers,train_loader,verify,verify_class,feat_names,epoch,lr):
    loss_detect=[]
    for i in workers:
        i.add_workers([j for j in workers if i!=j])

    criteria = nn.CrossEntropyLoss()
    criteria_loss = nn.CrossEntropyLoss()
    models, optims, regs = create_model_optim(workers, lr)

    for iter_round in range(epoch):
        n=0
        loc = workers[n]
        for imgs, labels in train_loader:
            if loc is not imgs.location:
                n += 1
                loc=workers[n]
            if n == 4:
                break
            optims[n].zero_grad()

            pred,dec_pred = models[n](imgs)

            loss=criteria(pred, labels)

            loss.backward()
            optims[n].step()
            loss = loss.get().data

        loss_detect.append(loss.item())
        print("final_epoch_loss: ",loss)

        send_models=[]

        if iter_round==(epoch-1):
           for idx, model in enumerate(models):
             send_models.append(explain(model,workers[idx],verify,idx))

    print(loss_detect)

    plt.plot([i+1 for i in range(epoch)], loss_detect)
    plt.show()

    enc_model=encrypt_share_gradient(send_models,workers)
    new_model=aggregate_gradient(enc_model,workers)

    return new_model



def evaluate(model, loader):
    f1list=[]
    labellist=[]
    problist=[]
    accuracy = 0
    for imgs, labels in loader:
        with th.no_grad():
            output,_=model.forward(imgs)
            ps = th.exp(output)
        top_p, top_class = ps.topk(1, dim=1)

        f1list.extend(top_class.detach().tolist())
        #f1list=[i[0] for i in f1list]
        labellist.extend(labels.detach().tolist())
        problist.extend(top_p.detach().tolist())

        prob = top_class == labels.view(*top_class.shape)
        prob = prob.float()
        accuracy += prob.mean().float()

    fpr, tpr, threshold = roc_curve(labellist,f1list)
    roc_auc = auc(fpr, tpr)

    print(f1_score(labellist,f1list))

    print("The accuracy of the model is {0}%".format((accuracy / len(loader)) * 100))

    #print(f1_score((top_class.view(*labels.shape)),np.array(loader.dataset).tolist()))

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



device=th.device("cuda" if th.cuda.is_available() else "cpu")

num_epochs = 5
num_classes = 2
batch_size = 2048
learning_rate = 0.05
#
hook=sf.TorchHook(th)
workers=create_workers(hook)
train_loader, test_loader, verify, verify_class, feat_name = generate("data/KDDTrain+.arff","data/KDDTest+.arff",workers,batch_size)

model=train_federated_model(workers,train_loader,verify,verify_class,feat_name,num_epochs,0.1)
evaluate(model,test_loader)





