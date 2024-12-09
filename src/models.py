import numpy as np

class RNN:
    def __init__(self,input_size,hidden_size,learning_rate,output_size,
                 weight_req_l1=0,weight_req_l2=0,dropout_rate=1):
        #hyperparams
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate
        #models params
        self.Wxh = np.random.randn(input_size,hidden_size)*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(hidden_size,output_size)*0.01 # hidden to output
        self.bx = np.zeros((1,hidden_size)) # input bias
        self.bh = np.zeros((1,hidden_size)) # hidden bias
        self.by = np.zeros((1,output_size)) # output bias
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
    
class LSTM: 
  
    def __init__(self,input_size,hidden_size,output_size):
        conc_input_size=input_size+hidden_size
        #forget gate weights
        self.Wf=np.random.randn(conc_input_size,hidden_size)*0.01
        self.bf = np.zeros((1, hidden_size))                       
        #input gate weights
        self.Wi=np.random.randn(conc_input_size,hidden_size)*0.01
        self.bi = np.zeros((1, hidden_size))                       
        #cell state candidate weights 
        self.Wc=np.random.randn(conc_input_size,hidden_size)*0.01
        self.bc = np.zeros((1, hidden_size))                       
        #output gate weighs
        self.Wo=np.random.randn(conc_input_size,hidden_size)*0.01
        self.bo = np.zeros((1, hidden_size))                       
        #output layer -->logits
        self.Wy=np.random.randn(hidden_size,output_size)*0.01
        self.by = np.zeros((1, output_size))  
    
    def forward(self,inputs,hprev,cprev):
        #dictionaries for all lstm states   
        states_and_gates = {
            'hidden_states': {},
            'cell_states': {},
            'concat_states': {},
            'forget_g': {},
            'input_g': {},
            'output_g': {},
            'cell_candidates': {},
            'logits': {}
        }
        
        for name, dict_storage in states_and_gates.items():
            setattr(self, name, dict_storage)

        #hidden state  and cell state  at t=0  
        self.hs[-1]=np.copy(hprev);self.cs[-1]=np.copy(cprev)
        for t in range(len(inputs)):
            self.concat_s[t]=np.concatenate([inputs[t],self.hs[t-1]],axis=1) 
            self.forget_g[t]=self.sigmoid(np.dot(self.concat_s[t],self.Wf)+self.bf) #forget gate 
            self.input_g[t]=self.sigmoid(np.dot(self.concat_s[t],self.Wi)+self.bi) # input gate 
            self.cell_candidates[t]=np.tanh(np.dot(self.concat_s[t],self.Wc)+self.bc)  #cell candidate 
            self.cs[t]=self.forget_g[t]*self.cs[t-1]+self.input_g[t]*self.cell_candidates[t] #new cell state 
            self.output_g[t]=self.sigmoid(np.dot(self.concat_s[t],self.Wo)+self.bo) #output gate 
            self.hs[t]=self.output_g[t]*np.tanh(self.cs[t])#hidden state 
            
            self.logits[t]=np.dot(self.hs[t],self.Wy)+self.by #output layer -->logits 
        
        #apply softmax after this 
        return self.logits

    def backward(self,dout,inputs):
        dWf,dbf,dWi,dbi,dWc,dbc,dWo,dbo,dWhy,dby,dh=self.init_gradients()
        dc_next = np.zeros_like(self.cell_states[0])
        dh_next = np.zeros_like(self.hidden_states[0])

        for t in reversed(range(len(inputs))):
            #output layer gradients   
            dWhy+=np.dot(self.hidden_states[t].T,dout[t]) #dy_dwhy 
            dby+=np.sum(dout[t],axis=0,keepdims=True) #dy_dby    
            dh+=np.dot(dout[t],self.Wy.T)+dh_next #dy_dh 
            
            tanh_c=np.tanh(self.cell_states[t]) 
            do=dh*tanh_c #dy_doutput_gate * dout(dh)
            do=self.output_g[t]*(1-self.output_g[t])*do #sigmoid derivative 
            #output gate gradients   
            d_conc_state=np.dot(do,self.Wo.T)
            dWo+=np.dot(self.concat_states[t].T,do)
            dbo+=np.sum(do,axis=0,keepdims=True)
            
            # cell state and cell candidate backpropagation 
            dc = np.copy(dc_next)
            dc+=dh * self.output_g[t]  * (1-tanh_c**2) #tanh backprop 
            dc_next = self.forget_g[t] * dc
            dc_candidate=dc*self.input_g[t]
            dc_candidate=(1-self.cell_candidates[t]**2)*dc_candidate #tanh backward * dout 

            d_conc_state += np.dot(dc_candidate, self.Wc.T)  # Backprop through matrix multiplication
            dWc += np.dot(self.concat_states[t].T, dc_candidate)  # Gradient for Wc
            dbc += np.sum(dc_candidate, axis=0, keepdims=True)  #
            #input gate backprop 
            d_input_gate=dc*self.cell_candidates[t] #from cell state equation above 
            d_input_gate=self.input_g[t]*(1-self.input_g[t])*d_input_gate #through sigmoid 
            # update w/b  for input gate
            dWi += np.dot(self.concat_states[t].T, d_input_gate)  # gradient for Wi
            dbi += np.sum(d_input_gate, axis=0, keepdims=True)    # gradient for bi
            # add to d_conc_state
            d_conc_state += np.dot(d_input_gate, self.Wi.T)  # gradient flowing back to concat_states
            #forget gate gradients 
            d_forget_gate = dc * self.cell_states[t-1]
            d_forget_gate=self.forget_g[t]*(1-self.forget_g[t])*d_forget_gate #through sigmoid 
            dWf += np.dot(self.concat_states[t].T, d_forget_gate)  # gradient for Wf
            dbf += np.sum(d_forget_gate, axis=0, keepdims=True)    # gradient for bf

            # Add forget gate contribution to d_conc_state
            d_conc_state += np.dot(d_forget_gate, self.Wf.T)  # gradient flowing back to concat_states 

            #TODO:compute concat state gradients 


    #init gradients with zero for backprop  
    def init_gradients(self):
        dWf=np.zeros_like(self.Wf);dbf=np.zeros_like(self.bf) #forget gates gradients  
        dWi=np.zeros_like(self.Wi);dbi=np.zeros_like(self.bi) #input gates gr
        dWc=np.zeros_like(self.Wc);dbc=np.zeros_like(self.bc) # cell state gr
        dWo=np.zeros_like(self.Wo);dbo=np.zeros_like(self.bo) # output gate gr 
        dWhy=np.zeros_like(self.Wy);dby=np.zeros_like(self.by)#logits layer gr
        dh=np.zeros_like(self.hidden_states[0])
        return dWf,dbf,dWi,dbi,dWc,dbc,dWo,dbo,dWhy,dby,dh

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
class Autoencoder:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        #encoder params
        self.W_enc = np.random.randn(input_size, hidden_size) * 0.01
        self.b_enc = np.zeros((1, hidden_size))
        #decoder params
        self.W_dec = np.random.randn(hidden_size, input_size) * 0.01
        self.b_dec = np.zeros((1, input_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self,X):
        #encoder 
        self.X=X
        self.z1=np.dot(X,self.W_enc)+self.b_enc
        self.a1=self.sigmoid(self.z1)
        #decoder 
        self.z2=np.dot(self.a1,self.W_dec)+self.b_dec
        self.a2=self.sigmoid(self.z2)

        return self.a2

    def backward(self,dout):

        dy_da2=self.a2*(1-self.a2) * dout #sigmoid derivative of a2 + chain rule  
        dy_da1=np.dot(dy_da2,self.W_dec.T)   
        dy_dw_dec=np.dot(self.a1.T,dy_da2) #dy_dweight_dec = matmul(a1(transposed),dy_da2) 
        dy_db_dec=np.sum(dy_da2,axis=0,keepdims=True) 
        
        dy_da1*=self.a1*(1-self.a1) #sigmoid der of a1 + chain rule 
        dy_dw_enc=np.dot(self.X.T,dy_da1)
        dy_db_enc=np.sum(dy_da1,axis=0,keepdims=True)
        
        return dy_dw_dec,dy_db_dec,dy_dw_enc,dy_db_enc

class MeanSquareError:
    
    def forward(self,logits,correct_labels):
        return np.mean(np.square((logits-correct_labels)))
    
    def backward(self,logits,correct_labels):
        return 2*(logits-correct_labels)/logits.shape[0]
        
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
