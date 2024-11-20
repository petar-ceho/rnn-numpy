import numpy as np
import random
import os
from src.models import RNN,CrossEntropyLoss
from helper import findFiles,readLines,n_letters,all_letters,TensorHelper

#train(80%),dev(10%),test(10%) no duplicate names 
def build_dataset(dataset):
    train_dataset,dev_dataset,test_dataset={},{},{}
    for key in dataset.keys():
        n1=int(0.8*len(dataset[key]))
        n2=int(0.9*len(dataset[key]))

        train_dataset.update({name: key for name in dataset[key][:n1]})
        dev_dataset.update({name: key for name in dataset[key][n1:n2]})
        test_dataset.update({name: key for name in dataset[key][n2:]})

    train_list,dev_list,test_list=list(train_dataset.items()),list(dev_dataset.items()),list(test_dataset.items())
    random.shuffle(train_list),random.shuffle(dev_list),random.shuffle(test_list)

    return dict(train_list),dict(dev_list),dict(test_list)

#TODO:implement here accuracy 
def evaluate(data,hprev,data_type,tensor_helper):
    lossi={}
    n=0
    for name,label in data.items():
        X=tensor_helper.line_to_tensor(name)
        Y=tensor_helper.label_to_onehot(label)

        logits,hprev=rnn_numpy.forward(X,hprev)
        probs,loss=cross_entropy.forward(logits,Y)
        lossi[n]=loss

        if n % 10000 == 0:
            print('loss with key  after iteration: ',n,loss,label)
            print(f'predicted label {all_categories[probs.argmax()]}: vs correct {all_categories[Y.argmax()]}')
        n+=1

    print(f'mean {data_type} loss after {n} iterations {sum(lossi.values())/n}')

if __name__ == '__main__':
    #download the dataset from here  https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    path='../data/names/*.txt'    
    
    category_lines = {} #dict of labels and features
    all_categories = [] #labels 

    for filename in findFiles(path): 
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    train,dev,test=build_dataset(category_lines)
   
    tensor_helper=TensorHelper(all_categories=all_categories,n_letters=n_letters,all_letters=all_letters)
    rnn_numpy=RNN(input_size=n_letters,hidden_size=50,learning_rate=0.001,output_size=len(all_categories),
              weight_req_l1=0.001,weight_req_l2=0.0001,dropout_rate=0.5)
    cross_entropy=CrossEntropyLoss(all_categories=all_categories)
    max_norm=5.0 #grazient norm clipping
    epochs=50
    n=0

    lossi,lossi_epochs,accuracy={},{},{}
    for i in range(epochs):
        #shuffle each epoch the training dataset to reduce overfitting
        train_list=list(train.items())
        random.shuffle(train_list)
        train=dict(train_list)
        loss_epoch,acc=0,0
        for name,label in train.items():

            X=tensor_helper.line_to_tensor(name)
            Y=tensor_helper.label_to_onehot(label)
            hprev=rnn_numpy.init_hidden() # init hidden state each sequence

            logits,hprev=rnn_numpy.forward(X,hprev)
            probs,loss=cross_entropy.forward(logits,Y)
            loss+=rnn_numpy.reqularization_loss()
            loss_epoch+=loss
            lossi[n]=loss
            acc+=int(probs.argmax()==Y.argmax())

            if n % 10000 == 0:
                print('loss with key  after iteration: ',n,loss,label)
                print(f'predicted label {all_categories[probs.argmax()]}: vs correct {all_categories[Y.argmax()]}')

            # backward pass
            dy=cross_entropy.backward(probs,Y)
            dWxh, dWhh, dWhy, dbh, dby, dbx=rnn_numpy.backward(dy,X)

            # gradient norm clipping
            total_norm=np.sqrt(np.sum(np.square(dWxh))+np.sum(np.square(dWhh))+np.sum(np.square(dWhy))+np.sum(np.square(dbx))+np.sum(np.square(dbh))+np.sum(np.square(dby)))
            if(total_norm>max_norm):
                clip_factor = max_norm / (total_norm + 1e-6)
                for param in [dWxh, dWhh, dWhy, dbh, dby, dbx]:
                    param*=clip_factor

            #  stohastic gradient descent TODO:implement ADAM
            for param,dparam in zip(
                [rnn_numpy.Wxh, rnn_numpy.Whh, rnn_numpy.Why, rnn_numpy.bh, rnn_numpy.by, rnn_numpy.bx],
                [dWxh, dWhh, dWhy, dbh, dby, dbx]):
                param+=-rnn_numpy.learning_rate*dparam
            n+=1

        lossi_epochs[i]=loss_epoch/len(train.items())
        accuracy[i]=acc/len(train.items())
    print(f'mean loss after {n} iterations {sum(lossi.values())/n}')
    print(f'mean accuracy after {n} iterations {sum(accuracy.values())/n}')


    evaluate(train,rnn_numpy.init_hidden(),'train',tensor_helper)
    evaluate(dev,rnn_numpy.init_hidden(),'dev',tensor_helper)
    evaluate(test,rnn_numpy.init_hidden(),'test',tensor_helper)
    
    #inference
    while True:
        # Prompt the user for input
        user_input = input("Enter a name (or type 'exit' to stop): ")
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        X=tensor_helper.line_to_tensor(user_input)
        logits,hprev=rnn_numpy.forward(X,rnn_numpy.init_hidden())
        probs,category=cross_entropy.sample(inputs=logits)

        print(f"Predicted category : {category} probability {probs.max()}")