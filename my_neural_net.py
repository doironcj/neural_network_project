
import random 
import math
import numpy as np

#this is my implementation of a neural network


#define the learning parameter 
η = .01
#define momentum
alpha = 0.0001

#define nodes per layer
npl = 50


#dot product 
def dot(x,y):
    sum = 0
    for i in range(0,len(x)):
        sum += x[i]*y[i]
    return sum
#vector adding function
def addv(x,y):
    r = []
    for i in range(0,len(x)):
        r.append(x[i]+y[i])
    return r
def compute_z(w,x,b):
    return dot(w,x)+b
def sigmoid(z):
  
    return 1/(1+np.exp(-z))
    #return (1/2)*math.tanh(z)+1
    
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
    #return (1/2)*(1-(sigmoid(z))**2)

# types 0 = bottom, 1 = head
class neuron:
    def __init__(self,t):
        self.w = []
        self.lower = []
        self.v = 0
        self.z = 0
        self.b = 0
        self.m = 0
        self.type = t
        self.values = []
        self.m = []
        self.wd = []
    def connect_layer(self,l):
        for i in range(0,len(l.neurons)):
            self.w.append(random.normalvariate(0,1)*math.sqrt(2/len(l.neurons)))
            self.m.append(0)
            self.wd.append(0)
        self.lower = l.neurons
        
    def add_values(self, vals):
        self.values = vals
        for i in range(0,len(self.values)):
            self.w.append(random.normalvariate(0,1)*math.sqrt(2/len(self.values)))
            self.m.append(0)
            self.wd.append(0)
    def change_values(self,vals):
        self.values = vals
    def compute(self):
        if(self.type == 0):
            self.z = compute_z(self.w,self.values,self.b)
            self.v = sigmoid(self.z)
            
            return self.v
        else:
            self.v = 0
            self.z = 0
            for n in range(0,len(self.lower)):
                self.z += self.lower[n].compute()*self.w[n]
            self.z += self.b
            self.v = sigmoid(self.z)
           
            return self.v
        
    def train_network(self,target):

        ds_dz = ((-target/self.v)+((1-target)/(1-self.v)))*sigmoid_prime(self.z)
        for i in range(0,len(self.lower)):
            self.lower[i].train_neuron(ds_dz,self.w[i])

        for i in range(0,len(self.w)):
            self.w[i] -= η*(ds_dz*self.lower[i].v+alpha*self.m[i])
            self.wd[i] += ds_dz*self.lower[i].v+alpha*self.m[i]
        self.b -= η*ds_dz
        
    def train_neuron(self,dsdz,upw):
        ds_dz = sigmoid_prime(self.z)*dsdz*upw
        for i in range(0,len(self.w)):
            self.w[i] -= η*(ds_dz*self.values[i]+alpha*self.m[i])
            self.wd[i] += ds_dz*self.values[i]+alpha*self.m[i]
        self.b -= η*ds_dz

    def print_network(self,f): 
        f.write(str(self.w))
        f.write("\n")
        if self.type == 1:
            for n in self.lower:
                n.print_network(f)
    def set_momentum(self,run_size):
        for i in range(0,len(self.m)):
            self.m[i] = self.wd[i]/run_size




class layer:
    def __init__(self,vals,neuron_number):
        self.neurons = []
        for i in range(0,neuron_number):
            n = neuron(0)
            n.add_values(vals)
            self.neurons.append(n)
    def change_values(self,vals):
        for n in self.neurons:
            n.change_values(vals)


#testing data sets
file_test_data = open("a2-test-data.txt","r")
file_test_label = open("a2-test-label.txt","r")
test_data  = []
test_label = []
for line in file_test_data:
    words = line.split()
    set = []
    for num in words:
        set.append(float(num)/100)#scale down
    test_data.append(set.copy())

for label in file_test_label:
    test_label.append(float(label))

# trainig data sets 
file_data = open("a2-train-data.txt","r")
file_label = open("a2-train-label.txt","r")
training_data  = []
training_label = []
for line in file_data:
    words = line.split()
    set = []
    for num in words:
        set.append(float(num)/100)#scale down 
    training_data.append(set.copy())

for label in file_label:
    training_label.append(float(label))

def train(head,bottom,sets,labels):
    num_right = 0
    avg_loss = 0
    order = list(range(0,len(labels)))
    random.shuffle(order)
    for i in order:
        bottom.change_values(sets[i])
        result = head.compute()
        target = (labels[i]+1)/2
       
        head.train_network(target)
        avg_loss += (1/2)*(result-target)**2
        if((result > 0.5 and target == 1) or(result < 0.5 and target ==  0.0)):
            num_right += 1
    head.set_momentum(len(labels))   
        
    print("average loss: {:.9f}".format(avg_loss/len(labels)))
    print("number correct: {:.9f}".format(num_right))
    return num_right
#check the accuracy
def check(head,bottom,sets,labels,f):
    num_right = 0
    
    for i in range(0,len(labels)):
        bottom.change_values(sets[i])
        result = head.compute()
        target = (labels[i]+1)/2
        head.train_network(target)
        if(result > 0.5):
            f.write("1 ")
        else:
            f.write("-1 ")
        if((result > 0.5 and target == 1) or(result < 0.5 and target ==  0.0)):
            num_right += 1
         
    print("total trials:")
    print(len(labels))
    print("total correct:")
    print(num_right)
    
#hidden layer
print("total test data entries:")
print(len(training_label))
bottom_layer = layer(training_data[0],npl)
head = neuron(1)
head.connect_layer(bottom_layer)
loss = 1
correct = 0
while(correct != 900):
    correct = train(head, bottom_layer, training_data,training_label)
    
   
f_prediction = open("predictions.txt","w")

check(head, bottom_layer, test_data,test_label,f_prediction)
f_prediction.close()

fout = open("network.txt","w")
fout.write(str(npl))
fout.write("\n")
head.print_network(fout) 
fout.close()







