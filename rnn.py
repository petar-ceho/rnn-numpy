import numpy as np

class RNN:

    def __init__(self,n_letters,hidden_size,learning_rate,target_length,
                 weight_req_l1=0,weight_req_l2=0,dropout_rate=1):
        #hyperparams
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate
        #models params
        self.Wxh = np.random.randn(n_letters,hidden_size)*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(hidden_size,target_length)*0.01 # hidden to output
        self.bx = np.zeros((1,hidden_size)) # input bias
        self.bh = np.zeros((1,hidden_size)) # hidden bias
        self.by = np.zeros((1,target_length)) # output bias
        self.weight_req_l1=weight_req_l1 #l1 reqularization  
        self.weight_req_l2=weight_req_l2 #l2 reqularization 
        self.dropout_rate=1-dropout_rate #percentage of neurons to keep  

    def forward(self,inputs,hprev):
        self.hs={}  #reset hidden state  memory each sequence 
        self.hs[-1]=np.copy(hprev) #hs[t-1] at t=0 

        if(self.dropout_rate>0): #variational dropouts for RNN(binary mask is the same across all sequence time steps)
            self.binary_mask=np.random.binomial(n=1,p=self.dropout_rate,size=hprev.shape)/self.dropout_rate
        for t in range(len(inputs)): #compute hidden state 
            self.hs[t]=np.tanh((np.dot(inputs[t],self.Wxh)+self.bx)+(np.dot(self.hs[t-1],self.Whh)+self.bh))
            if(self.dropout_rate>0): #if dropout rate is given  
                self.hs[t]*=self.binary_mask
        
        logits=np.dot(self.hs[list(self.hs.keys())[-1]],self.Why)+self.by #output layer 
        return logits,self.hs[list(self.hs.keys())[-1]] 
    
    def backward(self,dout,inputs):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbx, dbh= np.zeros_like(self.bx),np.zeros_like(self.bh)
        
        #output layer backpropagation  dy_dwhy and dy_dby and dy_dh
        dWhy=np.dot(self.hs[list(self.hs.keys())[-1]].T,dout) #dy_dwhy 
        dby=np.sum(dout,axis=0,keepdims=True) #dy_dby    
        dh=np.dot(dout,self.Why.T)   #dy_dh
    
        for t in reversed(range(len(inputs))): #BPTT
            if(self.dropout_rate>0): 
                dh*=self.binary_mask #dy_dh dropout layer
            dhraw=(1-self.hs[t]*self.hs[t])*dh #tanh backpropagation
            dh = np.dot(dhraw, self.Whh.T) #dy_dh for t-1    
            dbh+=np.sum(dhraw,axis=0,keepdims=True)#dhraw_dbh
            dbx+=np.sum(dhraw,axis=0,keepdims=True)#dhraw_dbx
            dWxh+=np.dot(inputs[t].T,dhraw)#dhraw_dwxh
            dWhh+=np.dot(self.hs[t-1].T,dhraw)#dhraw_dwhh
        
        # l2  backpropagation 
        if(self.weight_req_l2>0):
            dWxh+=2*self.weight_req_l2*self.Wxh
            dWhh+=2*self.weight_req_l2*self.Whh
            dWhy+=2*self.weight_req_l2*self.Why

        # l1  backpropagation
        if(self.weight_req_l1>0):
            dL1_dWxh,dL1_dWhh,dL1_dWhy=np.ones_like(self.Wxh),np.ones_like(self.Whh),np.ones_like(self.Why)
            dL1_dWxh[self.Wxh<0]=-1;dL1_dWhh[self.Whh<0]=-1;dL1_dWhy[self.Why<0]=-1
            dWxh+=self.weight_req_l1*dL1_dWxh;dWhh+=self.weight_req_l1*dL1_dWhh;dWhy+=self.weight_req_l1*dL1_dWhy
        

        return dWxh, dWhh, dWhy, dbh, dby, dbx

    
    def reqularization_loss(self):
        reqularization_loss=0
        
        if(self.weight_req_l1>0):
            reqularization_loss+=self.weight_req_l1*np.sum(np.abs(self.Wxh))
            reqularization_loss+=self.weight_req_l1*np.sum(np.abs(self.Whh))
            reqularization_loss+=self.weight_req_l1*np.sum(np.abs(self.Why))
        if(self.weight_req_l2>0):
            reqularization_loss+=self.weight_req_l2*np.sum(np.square(self.Wxh))
            reqularization_loss+=self.weight_req_l2*np.sum(np.square(self.Whh))
            reqularization_loss+=self.weight_req_l2*np.sum(np.square(self.Why))

        return reqularization_loss


    def init_hidden(self):
        return np.zeros((1,self.hidden_size))
    
#softmax+categorical cross entropy 
class CrossEntropyLoss: 

    def __init__(self,all_categories):
        self.all_categories=all_categories #labels
    
    # this will work if labels are one-hot and inputs are shape [1,N] 
    def forward(self,inputs,y_true):
        probs=np.exp(inputs)/np.sum(np.exp(inputs),axis=1) #softmax
        loss=-np.log(probs.reshape(-1)[y_true.argmax()]) #categorical cross entropy
        return probs,loss
    
    def sample(self,inputs): #inference
        probs=np.exp(inputs)/np.sum(np.exp(inputs),axis=1) #softmax
        return probs,self.all_categories[probs.argmax()]
    
    def backward(self,dout,y_true):
        dout[0][y_true.argmax()] -= 1 #backprop into y. 
        return dout
