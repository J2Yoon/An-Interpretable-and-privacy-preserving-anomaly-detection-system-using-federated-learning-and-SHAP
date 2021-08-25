# FederatedIDS

This project is for an interpretable privacy-preserving intrusion detection system(IDS) using federated learning and SHAP.

## Federated Learning
Federated Learning is the method, created by Google for privacy protection. This method protects privacy by sending only parameters to the server, not data.
There are 3 process in here:
1. Train each worker's model
2. Send the parameter to the server(Use encryption)
3. Server updated the parameter of model

```Python
#Create worker
workers=create_workers(hook)

# Train each worker
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
            

#encrypt and aggeregate model
enc_model=encrypt_share_gradient(send_models,workers)
new_model=aggregate_gradient(enc_model,workers)

```

##
SHAP is the interpretable method that calculated the importance of subsets of features.
This paper is original paper:
https://papers.nips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

Below picture is the explanation of IDS dataset in attack identification:
![instance_neg](https://user-images.githubusercontent.com/42733881/130741126-a25ac8d3-2e40-4ae2-b73b-281936d8f7a7.png)




